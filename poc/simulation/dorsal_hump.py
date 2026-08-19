"""Landmark-driven, non-authoritative dorsal hump reduction simulation."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from poc.logging_utils import get_logger
from poc.pipeline.geometry import geometry_identity

REQUIRED_LANDMARKS = {
    "nasion",
    "pronasale",
    "subnasale",
    "left_alare",
    "right_alare",
}
MIN_REDUCTION_MM = 0.0
MAX_REDUCTION_MM = 5.0
DORSAL_VAULT_SOLVER_ID = "coupled-vector-biharmonic-full-vault-v1"
LOWER_DORSUM_CEILING_START = 0.62
LOWER_DORSUM_BAND_START = 0.70
LOWER_DORSUM_CEILING_END = 0.82
MAX_LOWER_DORSUM_CORRECTION_FRACTION = 0.35
# Medial refinement remains mild even though it is solved together with the
# posterior full-vault field.
MAX_TRANSVERSE_MEDIAL_ADJUSTMENT_MM = 0.6


def _unit(vector: np.ndarray, label: str) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < 1e-9:
        raise ValueError(f"Cannot define {label} from coincident landmarks")
    return vector / length


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _single_mesh(path: Path):
    import trimesh

    loaded = trimesh.load_mesh(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) != 1:
            raise RuntimeError(f"Expected one mesh in {path}, found {len(loaded.geometry)}")
        loaded = next(iter(loaded.geometry.values()))
    return loaded


def _anatomical_frame(landmarks_mm: dict[str, np.ndarray]) -> dict[str, np.ndarray | float]:
    nasion = landmarks_mm["nasion"]
    pronasale = landmarks_mm["pronasale"]
    subnasale = landmarks_mm["subnasale"]
    vertical = _unit(subnasale - nasion, "nasion-to-subnasale axis")

    tip_vector = pronasale - nasion
    tip_longitudinal_mm = float(np.dot(tip_vector, vertical))
    anterior_raw = tip_vector - tip_longitudinal_mm * vertical
    anterior = _unit(anterior_raw, "anterior nasal projection axis")

    lateral = _unit(np.cross(vertical, anterior), "left-right axis")
    observed_left_to_right = landmarks_mm["right_alare"] - landmarks_mm["left_alare"]
    if float(np.dot(lateral, observed_left_to_right)) < 0.0:
        lateral *= -1.0
    anterior = _unit(np.cross(lateral, vertical), "orthogonal anterior axis")
    if float(np.dot(anterior, anterior_raw)) < 0.0:
        anterior *= -1.0
        lateral *= -1.0

    if tip_longitudinal_mm < 15.0:
        raise ValueError("Nasion, pronasale, and subnasale do not define a usable dorsal length")
    alar_width_mm = float(np.linalg.norm(landmarks_mm["right_alare"] - landmarks_mm["left_alare"]))
    if not np.isfinite(alar_width_mm) or alar_width_mm < 10.0:
        raise ValueError("Alar landmarks do not define a usable nasal width")
    roi_end_mm = min(0.82 * tip_longitudinal_mm, tip_longitudinal_mm - 6.0)
    if roi_end_mm < 12.0:
        raise ValueError("Landmarks do not leave a safe dorsum ROI before the nasal tip")
    half_width_mm = float(np.clip(0.28 * alar_width_mm, 6.0, 12.0))
    return {
        "origin": nasion,
        "vertical": vertical,
        "anterior": anterior,
        "lateral": lateral,
        "tip_longitudinal_mm": tip_longitudinal_mm,
        "roi_start_mm": 0.0,
        "roi_end_mm": roi_end_mm,
        "half_width_mm": half_width_mm,
        "alar_width_mm": alar_width_mm,
        "surface_depth_mm": 3.5,
    }


def _coordinates(vertices_mm: np.ndarray, frame: dict[str, Any]) -> tuple[np.ndarray, ...]:
    centered = vertices_mm - frame["origin"]
    longitudinal = centered @ frame["vertical"]
    lateral = centered @ frame["lateral"]
    anterior = centered @ frame["anterior"]
    return longitudinal, lateral, anterior


def _smooth_series(values: np.ndarray, sigma_bins: float = 1.6) -> np.ndarray:
    radius = max(2, int(np.ceil(3.0 * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _dorsal_profile(
    longitudinal: np.ndarray,
    lateral: np.ndarray,
    anterior: np.ndarray,
    frame: dict[str, Any],
    *,
    bin_count: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = float(frame["roi_start_mm"])
    end = float(frame["roi_end_mm"])
    strip_half_width = min(2.5, float(frame["half_width_mm"]) * 0.35)
    edges = np.linspace(start, end, bin_count + 1)
    centers = (edges[:-1] + edges[1:]) * 0.5
    profile = np.full(bin_count, np.nan, dtype=np.float64)
    for index in range(bin_count):
        active = (
            (longitudinal >= edges[index])
            & (longitudinal < edges[index + 1])
            & (np.abs(lateral) <= strip_half_width)
        )
        if np.count_nonzero(active) >= 2:
            profile[index] = float(np.percentile(anterior[active], 90.0))
    available = np.flatnonzero(np.isfinite(profile))
    if len(available) < max(12, bin_count // 3):
        raise RuntimeError(
            "The authoritative mesh has insufficient midline dorsal coverage for simulation"
        )
    profile = np.interp(np.arange(bin_count), available, profile[available])
    profile = _smooth_series(profile)

    normalized = (centers - start) / (end - start)
    proximal_anchor = (normalized >= 0.01) & (normalized <= 0.08)
    distal_anchor = (normalized >= 0.94) & (normalized <= 0.995)
    proximal_position = float(np.median(centers[proximal_anchor]))
    distal_position = float(np.median(centers[distal_anchor]))
    proximal_height = float(np.median(profile[proximal_anchor]))
    distal_height = float(np.median(profile[distal_anchor]))
    target_slope = (distal_height - proximal_height) / (distal_position - proximal_position)
    target = proximal_height + target_slope * (centers - proximal_position)
    return centers, profile, target


def _smootherstep01(value: np.ndarray) -> np.ndarray:
    """C2-continuous interpolation with zero slope and curvature at both ends."""
    clipped = np.clip(value, 0.0, 1.0)
    return clipped**3 * (clipped * (clipped * 6.0 - 15.0) + 10.0)


def _mesh_edges(faces: np.ndarray, vertex_count: int) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Simulation topology must contain Nx3 triangle indices")
    if len(faces) == 0 or np.min(faces) < 0 or np.max(faces) >= vertex_count:
        raise ValueError("Simulation topology contains no usable triangles or invalid indices")
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _connected_dorsal_component(
    candidate_mask: np.ndarray,
    edges: np.ndarray,
    seed_vertex: int,
) -> np.ndarray:
    """Keep only the connected candidate surface containing the detected apex."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    candidates = np.flatnonzero(candidate_mask)
    local_index = np.full(len(candidate_mask), -1, dtype=np.int64)
    local_index[candidates] = np.arange(len(candidates), dtype=np.int64)
    active_edges = edges[candidate_mask[edges[:, 0]] & candidate_mask[edges[:, 1]]]
    if not len(active_edges):
        raise RuntimeError("The nasal dorsum ROI contains no connected mesh edges")
    row = np.concatenate([local_index[active_edges[:, 0]], local_index[active_edges[:, 1]]])
    column = np.concatenate([local_index[active_edges[:, 1]], local_index[active_edges[:, 0]]])
    adjacency = coo_matrix(
        (np.ones(len(row), dtype=np.float64), (row, column)),
        shape=(len(candidates), len(candidates)),
    ).tocsr()
    _, labels = connected_components(adjacency, directed=False)
    seed_local = int(local_index[seed_vertex])
    if seed_local < 0:
        raise RuntimeError("The detected dorsal apex is outside the connected nasal ROI")
    connected = np.zeros_like(candidate_mask)
    connected[candidates[labels == labels[seed_local]]] = True
    return connected


def _laplacian_vector_deformation(
    *,
    faces: np.ndarray,
    roi_vertex_mask: np.ndarray,
    boundary_mask: np.ndarray,
    constraint_mask: np.ndarray,
    desired_components_mm: np.ndarray,
    maximum_posterior_mm: float,
    maximum_medial_mm: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one coupled posterior/lateral field over the connected dorsal vault."""
    from scipy.sparse import coo_matrix, diags, eye
    from scipy.sparse.linalg import spsolve

    edges = _mesh_edges(faces, len(roi_vertex_mask))
    roi_vertices = np.flatnonzero(roi_vertex_mask)
    local_index = np.full(len(roi_vertex_mask), -1, dtype=np.int64)
    local_index[roi_vertices] = np.arange(len(roi_vertices), dtype=np.int64)
    active_edges = edges[roi_vertex_mask[edges[:, 0]] & roi_vertex_mask[edges[:, 1]]]
    if not len(active_edges):
        raise RuntimeError("The connected nasal dorsum ROI contains no triangle edges")

    edge_u = local_index[active_edges[:, 0]]
    edge_v = local_index[active_edges[:, 1]]
    row = np.concatenate([edge_u, edge_v])
    column = np.concatenate([edge_v, edge_u])
    adjacency = coo_matrix(
        (np.ones(len(row), dtype=np.float64), (row, column)),
        shape=(len(roi_vertices), len(roi_vertices)),
    ).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse_sqrt_degree = 1.0 / np.sqrt(np.maximum(degree, 1.0))
    normalized_laplacian = eye(len(roi_vertices), format="csr") - (
        diags(inverse_sqrt_degree) @ adjacency @ diags(inverse_sqrt_degree)
    )

    local_boundary = boundary_mask[roi_vertices]
    local_constraints = constraint_mask[roi_vertices] & ~local_boundary
    if np.count_nonzero(local_constraints) < 3:
        raise RuntimeError("Too few dorsal vertices are available as deformation constraints")
    if np.count_nonzero(local_boundary) < 3:
        raise RuntimeError("Too few ROI boundary vertices are available to protect nearby anatomy")

    desired_components_mm = np.asarray(desired_components_mm, dtype=np.float64)
    if desired_components_mm.shape != (len(roi_vertex_mask), 2):
        raise ValueError("Dorsal-vault constraints must contain posterior/lateral components")

    # One matrix and one vector-valued right-hand side couple the complete
    # dorsal vault. Ridge, slopes, and sidewalls are constraints on the same
    # smooth sheet; there is no independently deformed center strip.
    constraint_weight = np.zeros(len(roi_vertices), dtype=np.float64)
    constraint_target = np.zeros((len(roi_vertices), 2), dtype=np.float64)
    constraint_weight[local_constraints] = 2500.0
    constraint_target[local_constraints] = desired_components_mm[roi_vertices[local_constraints]]
    constraint_weight[local_boundary] = 10000.0
    system = (
        normalized_laplacian.T @ normalized_laplacian
        + diags(constraint_weight)
        + eye(len(roi_vertices), format="csr") * 1e-9
    ).tocsc()
    local_deformation = np.asarray(
        spsolve(system, constraint_weight[:, None] * constraint_target), dtype=np.float64
    )
    if not np.all(np.isfinite(local_deformation)):
        raise RuntimeError("Laplacian dorsal deformation did not converge to finite positions")
    local_deformation[:, 0] = np.clip(local_deformation[:, 0], 0.0, maximum_posterior_mm)
    local_deformation[:, 1] = np.clip(
        local_deformation[:, 1], -maximum_medial_mm, maximum_medial_mm
    )
    local_deformation[local_boundary] = 0.0
    displacement_components_mm = np.zeros((len(roi_vertex_mask), 2), dtype=np.float64)
    displacement_components_mm[roi_vertices] = local_deformation
    displacement_components_mm[np.abs(displacement_components_mm) < 1e-8] = 0.0

    edge_change = np.linalg.norm(
        displacement_components_mm[active_edges[:, 0]]
        - displacement_components_mm[active_edges[:, 1]],
        axis=1,
    )
    diagnostics = {
        "solver_id": DORSAL_VAULT_SOLVER_ID,
        "method": "coupled_vector_biharmonic_dorsal_vault",
        "connected_roi_vertex_count": len(roi_vertices),
        "constraint_vertex_count": int(np.count_nonzero(local_constraints)),
        "fixed_boundary_vertex_count": int(np.count_nonzero(local_boundary)),
        "maximum_neighbor_displacement_change_mm": round(float(np.max(edge_change)), 6),
        "p95_neighbor_displacement_change_mm": round(float(np.percentile(edge_change, 95)), 6),
    }
    return displacement_components_mm, diagnostics


def _compute_dorsal_hump_deformation(
    vertices_m: np.ndarray,
    landmarks_mm: dict[str, np.ndarray],
    reduction_mm: float,
    faces: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
]:
    """Compute deformation plus internal ROI/profile diagnostics without mutating inputs."""
    reduction_mm = float(reduction_mm)
    if not np.isfinite(reduction_mm) or not MIN_REDUCTION_MM <= reduction_mm <= MAX_REDUCTION_MM:
        raise ValueError(
            f"Dorsal hump reduction must be between {MIN_REDUCTION_MM:.1f} and "
            f"{MAX_REDUCTION_MM:.1f} mm"
        )
    vertices_m = np.asarray(vertices_m, dtype=np.float64)
    if vertices_m.ndim != 2 or vertices_m.shape[1] != 3 or not np.all(np.isfinite(vertices_m)):
        raise ValueError("Simulation requires finite Nx3 authoritative vertices")
    missing = sorted(REQUIRED_LANDMARKS - landmarks_mm.keys())
    if missing:
        raise ValueError(f"Missing landmarks required for dorsal simulation: {missing}")
    normalized_landmarks = {
        name: np.asarray(value, dtype=np.float64) for name, value in landmarks_mm.items()
    }
    frame = _anatomical_frame(normalized_landmarks)
    vertices_mm = vertices_m * 1000.0
    longitudinal, lateral, anterior = _coordinates(vertices_mm, frame)

    roi_metadata: dict[str, Any] = {
        "definition": "landmark_driven_midline_dorsum_nasion_to_inferred_supratip",
        "longitudinal_start_mm_from_nasion": float(frame["roi_start_mm"]),
        "longitudinal_end_mm_from_nasion": round(float(frame["roi_end_mm"]), 6),
        "inferred_supratip_fraction_of_nasion_to_tip": 0.82,
        "lateral_half_width_mm": round(float(frame["half_width_mm"]), 6),
        "surface_depth_mm": float(frame["surface_depth_mm"]),
        "world_axes": {
            "origin_nasion_mm": np.asarray(frame["origin"]).round(8).tolist(),
            "superior_to_inferior": np.asarray(frame["vertical"]).round(10).tolist(),
            "patient_left_to_right": np.asarray(frame["lateral"]).round(10).tolist(),
            "posterior_to_anterior": np.asarray(frame["anterior"]).round(10).tolist(),
        },
        "protected_regions": [
            "nasal_tip",
            "columella",
            "nostrils",
            "alar_rims",
            "upper_lip",
            "cheeks",
        ],
    }
    centers, profile, reference_profile = _dorsal_profile(
        longitudinal,
        lateral,
        anterior,
        frame,
    )
    start = float(frame["roi_start_mm"])
    end = float(frame["roi_end_mm"])
    normalized_centers = (centers - start) / (end - start)
    convex_excess = np.maximum(profile - reference_profile, 0.0)
    apex_search = (normalized_centers >= 0.08) & (normalized_centers <= 0.62)
    apex_candidates = np.flatnonzero(apex_search)
    apex_index = int(apex_candidates[np.argmax(convex_excess[apex_search])])
    available_hump_mm = float(convex_excess[apex_index])
    apex_normalized = float(normalized_centers[apex_index])
    if reduction_mm > 0.0 and available_hump_mm <= 1e-6:
        raise RuntimeError("No positive upper/mid-dorsal convexity was found for hump reduction")
    applied_reduction_mm = min(reduction_mm, available_hump_mm)
    # Do not drive the profile behind its nasion-to-supratip reference.  The
    # requested value is a ceiling; the detectable convexity is the anatomical
    # ceiling.  This prevents an oversized slider request from creating a new
    # sagittal scoop.
    upper_fade = _smootherstep01(normalized_centers / 0.08)
    distal_support_end = min(0.82, max(0.70, apex_normalized + 0.28))
    distal_fade = 1.0 - _smootherstep01(
        (normalized_centers - apex_normalized)
        / max(distal_support_end - apex_normalized, 1e-9)
    )
    upper_mid_window = upper_fade * np.where(
        normalized_centers <= apex_normalized,
        1.0,
        distal_fade,
    )
    applied_convexity_fraction = (
        applied_reduction_mm / available_hump_mm if available_hump_mm > 1e-9 else 0.0
    )
    # Reduce the complete superior shoulder and apex by one fraction, then fade
    # smoothly through the mid dorsum. This deliberately avoids carrying the
    # peak correction into the lower dorsum and supratip.
    unconstrained_centerline_reduction = np.minimum(
        convex_excess * applied_convexity_fraction * upper_mid_window,
        applied_reduction_mm,
    )
    # A separate lower convexity can be larger than the selected upper/mid
    # apex. Apply a C2 ceiling only where the normal anatomical fade would
    # exceed the lower-dorsum safety limit: full peak permission through 0.62,
    # at most 35% at 0.70, and zero at the inferred supratip boundary 0.82.
    proximal_lower_ceiling = 1.0 - (
        1.0 - MAX_LOWER_DORSUM_CORRECTION_FRACTION
    ) * _smootherstep01(
        (normalized_centers - LOWER_DORSUM_CEILING_START)
        / (LOWER_DORSUM_BAND_START - LOWER_DORSUM_CEILING_START)
    )
    distal_lower_ceiling = MAX_LOWER_DORSUM_CORRECTION_FRACTION * (
        1.0
        - _smootherstep01(
            (normalized_centers - LOWER_DORSUM_BAND_START)
            / (LOWER_DORSUM_CEILING_END - LOWER_DORSUM_BAND_START)
        )
    )
    lower_dorsum_ceiling_fraction = np.where(
        normalized_centers <= LOWER_DORSUM_BAND_START,
        proximal_lower_ceiling,
        distal_lower_ceiling,
    )
    centerline_reduction = np.minimum.reduce(
        (
            unconstrained_centerline_reduction,
            applied_reduction_mm * lower_dorsum_ceiling_fraction,
            convex_excess,
        )
    )
    target_profile = profile - centerline_reduction
    maximum_new_below_reference_mm = float(
        np.max(
            np.maximum(reference_profile - target_profile, 0.0)
            - np.maximum(reference_profile - profile, 0.0)
        )
    )

    source_profile_at_vertex = np.interp(
        longitudinal,
        centers,
        profile,
        left=profile[0],
        right=profile[-1],
    )
    desired_profile_displacement_mm = np.interp(
        longitudinal,
        centers,
        centerline_reduction,
        left=0.0,
        right=0.0,
    )
    profile_core_half_width_mm = min(2.5, 0.35 * float(frame["half_width_mm"]))
    vault_half_width_mm = min(
        22.0,
        0.56 * float(frame["alar_width_mm"]),
        2.00 * float(frame["half_width_mm"]),
    )
    bridge_core_half_width_mm = 0.68 * vault_half_width_mm
    within_longitudinal = (longitudinal > start) & (longitudinal < end)
    surface_gap_mm = source_profile_at_vertex - anterior
    surface_depth_limit_mm = max(8.0, float(frame["surface_depth_mm"]), applied_reduction_mm + 4.0)
    surface_band = (surface_gap_mm >= -1.5) & (surface_gap_mm <= surface_depth_limit_mm)
    lateral_radius = np.abs(lateral) / max(vault_half_width_mm, 1e-9)
    surface_radius = np.clip(np.maximum(surface_gap_mm, 0.0) / surface_depth_limit_mm, 0.0, 1.0)
    vault_radius = np.maximum(lateral_radius, 0.80 * np.sqrt(surface_radius))
    candidate_roi_mask = within_longitudinal & (vault_radius < 1.0) & surface_band

    # One explicit transverse vault field: the ridge receives delta(s), dorsal
    # slopes retain most of it, and the sidewalls taper smoothly to the fixed
    # anatomical perimeter.  There is no special center-strip permission mask.
    inner_vault_weight = 1.0 - 0.06 * _smootherstep01(vault_radius / 0.68)
    outer_vault_weight = 0.94 * (
        1.0 - _smootherstep01((vault_radius - 0.68) / (1.0 - 0.68))
    )
    transverse_vault_weight = np.where(
        vault_radius <= 0.68,
        inner_vault_weight,
        outer_vault_weight,
    )
    transverse_vault_weight[~candidate_roi_mask] = 0.0
    longitudinal_correction_weight = (
        desired_profile_displacement_mm / applied_reduction_mm
        if applied_reduction_mm > 0.0
        else np.zeros(len(vertices_m), dtype=np.float64)
    )
    desired_posterior_mm = desired_profile_displacement_mm * transverse_vault_weight

    # Mild medialization is part of the same vector target. It is zero on the
    # ridge, peaks across the slopes/sidewalls, and returns to zero at the vault
    # boundary, so the solve cannot leave stationary lateral rails.
    maximum_narrowing_mm = min(MAX_TRANSVERSE_MEDIAL_ADJUSTMENT_MM, 0.1 * applied_reduction_mm)
    transverse_narrowing_weight = _smootherstep01(vault_radius / 0.35) * (
        1.0 - _smootherstep01((vault_radius - 0.72) / 0.28)
    )
    desired_lateral_shift_mm = (
        -np.sign(lateral)
        * maximum_narrowing_mm
        * longitudinal_correction_weight
        * transverse_narrowing_weight
    )
    desired_components_mm = np.column_stack(
        [desired_posterior_mm, desired_lateral_shift_mm]
    )

    if reduction_mm == 0.0:
        roi_vertex_mask = candidate_roi_mask
        displacement_components_mm = np.zeros((len(vertices_m), 2), dtype=np.float64)
        deformation_solver = {
            "solver_id": DORSAL_VAULT_SOLVER_ID,
            "method": "identity_zero_reduction",
            "connected_roi_vertex_count": int(np.count_nonzero(roi_vertex_mask)),
            "constraint_vertex_count": 0,
            "fixed_boundary_vertex_count": 0,
            "maximum_neighbor_displacement_change_mm": 0.0,
            "p95_neighbor_displacement_change_mm": 0.0,
        }
    elif faces is None:
        # Compatibility fallback only. Production always supplies topology and
        # uses the coupled sparse solve below.
        roi_vertex_mask = candidate_roi_mask
        displacement_components_mm = desired_components_mm * roi_vertex_mask[:, None]
        displacement_components_mm[np.abs(displacement_components_mm) < 1e-8] = 0.0
        deformation_solver = {
            "solver_id": DORSAL_VAULT_SOLVER_ID,
            "method": "analytic_full_vault_target_no_topology",
            "connected_roi_vertex_count": int(np.count_nonzero(roi_vertex_mask)),
            "constraint_vertex_count": int(np.count_nonzero(roi_vertex_mask)),
            "fixed_boundary_vertex_count": 0,
            "maximum_neighbor_displacement_change_mm": None,
            "p95_neighbor_displacement_change_mm": None,
        }
    else:
        edges = _mesh_edges(faces, len(vertices_m))
        apex_vertex_score = (
            np.abs(longitudinal - centers[apex_index])
            + 2.0 * np.abs(lateral)
            + np.abs(anterior - profile[apex_index])
        )
        apex_vertex_score[~candidate_roi_mask] = np.inf
        if not np.any(np.isfinite(apex_vertex_score)):
            raise RuntimeError("The detected hump apex has no vertices in the nasal surface band")
        apex_vertex = int(np.argmin(apex_vertex_score))
        roi_vertex_mask = _connected_dorsal_component(candidate_roi_mask, edges, apex_vertex)
        edge_has_outside_neighbor = np.zeros(len(vertices_m), dtype=bool)
        crossing_edges = edges[roi_vertex_mask[edges[:, 0]] != roi_vertex_mask[edges[:, 1]]]
        if len(crossing_edges):
            inside_endpoints = np.where(
                roi_vertex_mask[crossing_edges[:, 0]], crossing_edges[:, 0], crossing_edges[:, 1]
            )
            edge_has_outside_neighbor[inside_endpoints] = True
        normalized_vertex_position = (longitudinal - start) / (end - start)
        boundary_mask = roi_vertex_mask & (
            edge_has_outside_neighbor
            | (normalized_vertex_position <= 0.04)
            | (normalized_vertex_position >= 0.96)
            | (vault_radius >= 0.94)
        )
        vault_constraint_mask = roi_vertex_mask & ~boundary_mask
        displacement_components_mm, deformation_solver = _laplacian_vector_deformation(
            faces=np.asarray(faces, dtype=np.int64),
            roi_vertex_mask=roi_vertex_mask,
            boundary_mask=boundary_mask,
            constraint_mask=vault_constraint_mask,
            desired_components_mm=desired_components_mm,
            maximum_posterior_mm=applied_reduction_mm,
            maximum_medial_mm=maximum_narrowing_mm,
        )

    posterior_displacement_mm = displacement_components_mm[:, 0]
    lateral_shift_mm = displacement_components_mm[:, 1]
    # Numerical smoothing must never move either side farther away from the
    # midline or allow a vertex to cross it.
    lateral_shift_mm[lateral * lateral_shift_mm > 0.0] = 0.0
    lateral_shift_mm = np.sign(lateral_shift_mm) * np.minimum(
        np.abs(lateral_shift_mm), 0.45 * np.abs(lateral)
    )
    target_vertices_m = (
        vertices_m
        - desired_posterior_mm[:, None] * np.asarray(frame["anterior"])[None, :] / 1000.0
        + desired_lateral_shift_mm[:, None] * np.asarray(frame["lateral"])[None, :] / 1000.0
    )
    simulated = (
        vertices_m
        - posterior_displacement_mm[:, None] * np.asarray(frame["anterior"])[None, :] / 1000.0
        + lateral_shift_mm[:, None] * np.asarray(frame["lateral"])[None, :] / 1000.0
    )
    displacement_mm = np.hypot(posterior_displacement_mm, lateral_shift_mm)
    roi_metadata["profile_model"] = {
        "observed_profile": "smoothed 90th-percentile midline anterior envelope",
        "reference_profile": "straight chord through robust nasion and supratip anchors",
        "target_profile": (
            "upper/mid convex-excess correction with C2 anatomical fades and a no-scoop "
            "nasion-to-supratip reference constraint"
        ),
        "convex_hump_only": True,
        "available_hump_height_mm": round(available_hump_mm, 6),
        "available_hump_height_is_clinical_measurement": False,
        "available_hump_height_interpretation": (
            "algorithmic convex excess above the simulation target profile"
        ),
        "requested_peak_policy": (
            "slider value is an upper bound; correction stops at the no-scoop reference profile"
        ),
        "requested_peak_reduction_mm": round(reduction_mm, 6),
        "applied_peak_reduction_mm": round(applied_reduction_mm, 6),
        "applied_upper_mid_convexity_fraction": round(applied_convexity_fraction, 6),
        "distal_support_end_normalized": round(distal_support_end, 6),
        "lower_dorsum_ceiling_normalized": [
            LOWER_DORSUM_CEILING_START,
            LOWER_DORSUM_BAND_START,
            LOWER_DORSUM_CEILING_END,
        ],
        "maximum_lower_dorsum_correction_fraction": (
            MAX_LOWER_DORSUM_CORRECTION_FRACTION
        ),
        "limited_by_detected_convexity": applied_reduction_mm < reduction_mm,
        "maximum_new_below_reference_mm": round(maximum_new_below_reference_mm, 9),
    }
    roi_metadata["profile_core_half_width_mm"] = round(profile_core_half_width_mm, 6)
    roi_metadata["transverse_bridge_core_half_width_mm"] = round(bridge_core_half_width_mm, 6)
    roi_metadata["profile_deformation_half_width_mm"] = round(vault_half_width_mm, 6)
    roi_metadata["surface_depth_limit_mm"] = round(surface_depth_limit_mm, 6)
    roi_metadata["landmark_estimated_dorsal_half_width_mm"] = roi_metadata["lateral_half_width_mm"]
    roi_metadata["lateral_half_width_mm"] = round(vault_half_width_mm, 6)
    roi_metadata["deformation_solver"] = deformation_solver
    roi_metadata["pointwise_envelope_clipping"] = False
    apex_level_mask = roi_vertex_mask & (
        np.abs(longitudinal - centers[apex_index]) <= max(1.0, 0.035 * (end - start))
    )

    def displacement_band(minimum_radius: float, maximum_radius: float) -> dict[str, Any]:
        band_mask = (
            apex_level_mask
            & (vault_radius >= minimum_radius)
            & (vault_radius < maximum_radius)
        )
        values = posterior_displacement_mm[band_mask]
        return {
            "normalized_radius": [minimum_radius, maximum_radius],
            "vertex_count": int(np.count_nonzero(band_mask)),
            "median_posterior_displacement_mm": round(
                float(np.median(values)) if len(values) else 0.0, 6
            ),
            "fraction_moved_over_0_1_mm": round(
                float(np.mean(values > 0.1)) if len(values) else 0.0, 6
            ),
        }

    ridge_band = displacement_band(0.0, 0.25)
    slope_band = displacement_band(0.25, 0.60)
    sidewall_band = displacement_band(0.60, 0.85)
    ridge_median = float(ridge_band["median_posterior_displacement_mm"])
    slope_to_ridge_ratio = (
        float(slope_band["median_posterior_displacement_mm"]) / ridge_median
        if ridge_median > 1e-9
        else 0.0
    )
    sidewall_to_ridge_ratio = (
        float(sidewall_band["median_posterior_displacement_mm"]) / ridge_median
        if ridge_median > 1e-9
        else 0.0
    )
    roi_metadata["vault_displacement_coverage"] = {
        "sampling_level": "detected_hump_apex",
        "ridge": ridge_band,
        "dorsal_slopes": slope_band,
        "sidewalls": sidewall_band,
        "slope_to_ridge_median_ratio": round(slope_to_ridge_ratio, 6),
        "sidewall_to_ridge_median_ratio": round(sidewall_to_ridge_ratio, 6),
        "center_strip_pattern_detected": (
            slope_to_ridge_ratio < 0.75 or sidewall_to_ridge_ratio < 0.45
        ),
    }
    roi_metadata["transverse_model"] = {
        "cross_section_behavior": (
            "one coupled ridge-slope-sidewall target across the connected dorsal vault"
        ),
        "ridge_target_weight": 1.0,
        "slope_target_weight_at_normalized_radius_0_5": round(
            float(1.0 - 0.06 * _smootherstep01(np.asarray([0.5 / 0.68]))[0]), 6
        ),
        "sidewall_target_weight_at_normalized_radius_0_8": round(
            float(0.94 * (1.0 - _smootherstep01(np.asarray([(0.8 - 0.68) / 0.32]))[0])),
            6,
        ),
        "zero_weight_normalized_radius": 1.0,
        "requested_maximum_medial_adjustment_mm": round(maximum_narrowing_mm, 6),
        "actual_maximum_medial_adjustment_mm": round(float(np.max(np.abs(lateral_shift_mm))), 6),
        "sidewall_blending": (
            "posterior and medial components solved together with a fixed anatomical perimeter"
        ),
        "coupled_single_solve": True,
        "entire_vault_is_constraint_surface": True,
        "left_right_symmetrization": False,
    }
    apex_point_mm = (
        np.asarray(frame["origin"])
        + centers[apex_index] * np.asarray(frame["vertical"])
        + profile[apex_index] * np.asarray(frame["anterior"])
    )
    roi_metadata["detected_hump_apex"] = {
        "profile_bin_index": apex_index,
        "normalized_nasion_to_supratip": round(apex_normalized, 6),
        "longitudinal_mm_from_nasion": round(float(centers[apex_index]), 6),
        "source_anterior_mm": round(float(profile[apex_index]), 6),
        "reference_anterior_mm": round(float(reference_profile[apex_index]), 6),
        "target_anterior_mm": round(float(target_profile[apex_index]), 6),
        "outward_convexity_mm": round(available_hump_mm, 6),
        "world_position_mm": apex_point_mm.round(6).tolist(),
        "search_band_normalized": [0.08, 0.62],
        "is_clinical_measurement": False,
    }
    roi_metadata["candidate_vertex_count"] = int(np.count_nonzero(roi_vertex_mask))
    final_longitudinal, final_lateral, final_anterior = _coordinates(simulated * 1000.0, frame)
    _, final_profile, _ = _dorsal_profile(
        final_longitudinal,
        final_lateral,
        final_anterior,
        frame,
    )
    profile_diagnostic = {
        "longitudinal_mm": centers,
        "source_anterior_mm": profile,
        "reference_anterior_mm": reference_profile,
        "target_anterior_mm": target_profile,
        "final_anterior_mm": final_profile,
        "simulated_anterior_mm": final_profile,
        "delta_mm": centerline_reduction,
        "centerline_reduction_mm": centerline_reduction,
        "posterior_displacement_mm": posterior_displacement_mm,
        "lateral_shift_mm": lateral_shift_mm,
        "transverse_vault_weight": transverse_vault_weight,
    }
    diagnostic_levels = {
        "radix_nasion": 0.0,
        "upper_dorsum": 0.18,
        "hump_apex": apex_normalized,
        "mid_dorsum": 0.55,
        "lower_dorsum": 0.72,
        "supratip": 0.88,
    }
    profile_points: dict[str, Any] = {}
    for name, normalized in diagnostic_levels.items():
        level_mm = start + float(normalized) * (end - start)
        source_value = float(np.interp(level_mm, centers, profile))
        reference_value = float(np.interp(level_mm, centers, reference_profile))
        target_value = float(np.interp(level_mm, centers, target_profile))
        final_value = float(np.interp(level_mm, centers, final_profile))
        point_convex_excess_mm = max(source_value - reference_value, 0.0)
        point_requested_displacement_mm = source_value - target_value
        profile_points[name] = {
            "normalized_nasion_to_supratip": round(float(normalized), 6),
            "longitudinal_mm_from_nasion": round(level_mm, 6),
            "source_anterior_mm": round(source_value, 6),
            "reference_anterior_mm": round(reference_value, 6),
            "target_anterior_mm": round(target_value, 6),
            "final_anterior_mm": round(final_value, 6),
            "source_convex_excess_mm": round(point_convex_excess_mm, 6),
            "requested_posterior_displacement_mm": round(point_requested_displacement_mm, 6),
            "requested_convexity_reduction_fraction": round(
                float(
                    np.clip(
                        point_requested_displacement_mm / point_convex_excess_mm
                        if point_convex_excess_mm > 1e-9
                        else 0.0,
                        0.0,
                        1.0,
                    )
                ),
                6,
            ),
            "final_posterior_displacement_mm": round(source_value - final_value, 6),
            "final_target_error_mm": round(final_value - target_value, 6),
        }
    tip_anterior_mm = float(
        np.dot(normalized_landmarks["pronasale"] - np.asarray(frame["origin"]), frame["anterior"])
    )
    profile_points["pronasale"] = {
        "normalized_nasion_to_supratip": None,
        "longitudinal_mm_from_nasion": round(float(frame["tip_longitudinal_mm"]), 6),
        "source_anterior_mm": round(tip_anterior_mm, 6),
        "reference_anterior_mm": None,
        "target_anterior_mm": round(tip_anterior_mm, 6),
        "final_anterior_mm": round(tip_anterior_mm, 6),
        "source_convex_excess_mm": 0.0,
        "requested_posterior_displacement_mm": 0.0,
        "requested_convexity_reduction_fraction": 0.0,
        "final_posterior_displacement_mm": 0.0,
        "final_target_error_mm": 0.0,
    }
    roi_metadata["profile_point_diagnostics"] = profile_points
    return (
        simulated,
        displacement_mm,
        roi_metadata,
        roi_vertex_mask,
        profile_diagnostic,
        target_vertices_m,
    )


def compute_dorsal_hump_deformation(
    vertices_m: np.ndarray,
    landmarks_mm: dict[str, np.ndarray],
    reduction_mm: float,
    faces: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return simulated vertices and displacement magnitudes without mutating input arrays."""
    simulated, displacement_mm, roi_metadata, _, _, _ = _compute_dorsal_hump_deformation(
        vertices_m,
        landmarks_mm,
        reduction_mm,
        faces,
    )
    return simulated, displacement_mm, roi_metadata


def _positive_median(values: np.ndarray, threshold: float = 1e-6) -> float:
    selected = values[values > threshold]
    return float(np.median(selected)) if len(selected) else 0.0


def _export_roi_ply(
    output_path: Path,
    vertices_m: np.ndarray,
    faces: np.ndarray,
    roi_vertex_mask: np.ndarray,
    displacement_mm: np.ndarray,
) -> dict[str, int]:
    import open3d as o3d

    roi_faces = faces[np.all(roi_vertex_mask[faces], axis=1)]
    if not len(roi_faces):
        raise RuntimeError("The selected nasal dorsum ROI contains no complete triangles")
    used_vertices = np.unique(roi_faces)
    remap = np.full(len(vertices_m), -1, dtype=np.int64)
    remap[used_vertices] = np.arange(len(used_vertices), dtype=np.int64)
    roi_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices_m[used_vertices]),
        o3d.utility.Vector3iVector(remap[roi_faces].astype(np.int32)),
    )
    peak = max(float(np.max(displacement_mm[used_vertices])), 1e-9)
    intensity = np.clip(displacement_mm[used_vertices] / peak, 0.0, 1.0)
    colors = np.column_stack(
        [
            0.15 + 0.85 * intensity,
            0.2 * np.ones_like(intensity),
            1.0 - 0.85 * intensity,
        ]
    )
    roi_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    roi_mesh.compute_vertex_normals()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output_path), roi_mesh, write_ascii=False):
        raise RuntimeError(f"Failed to write nasal ROI diagnostic: {output_path}")
    return {
        "vertex_count": len(used_vertices),
        "triangle_count": len(roi_faces),
    }


def _export_moved_vertices_ply(
    output_path: Path,
    vertices_m: np.ndarray,
    displacement_mm: np.ndarray,
) -> int:
    import open3d as o3d

    moved = displacement_mm > 1e-6
    if not np.any(moved):
        return 0
    peak = max(float(np.max(displacement_mm[moved])), 1e-9)
    intensity = np.clip(displacement_mm[moved] / peak, 0.0, 1.0)
    colors = np.column_stack(
        [
            0.15 + 0.85 * intensity,
            0.15 * np.ones_like(intensity),
            1.0 - 0.85 * intensity,
        ]
    )
    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices_m[moved]))
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    if not o3d.io.write_point_cloud(str(output_path), point_cloud, write_ascii=False):
        raise RuntimeError(f"Failed to write moved-vertex diagnostic: {output_path}")
    return int(np.count_nonzero(moved))


def _vertex_normals(vertices_m: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute area-weighted unit normals directly from the current geometry."""
    triangles = vertices_m[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    normals = np.zeros_like(vertices_m, dtype=np.float64)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    if not np.all(valid):
        raise RuntimeError("The simulation mesh contains vertices without usable normals")
    return normals


def _sample_cross_section(
    longitudinal: np.ndarray,
    lateral: np.ndarray,
    anterior: np.ndarray,
    *,
    level_mm: float,
    half_width_mm: float,
    longitudinal_window_mm: float,
    bin_count: int = 49,
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(-half_width_mm, half_width_mm, bin_count + 1)
    centers = (edges[:-1] + edges[1:]) * 0.5
    section = np.full(bin_count, np.nan, dtype=np.float64)
    near_level = np.abs(longitudinal - level_mm) <= longitudinal_window_mm
    for index in range(bin_count):
        selected = near_level & (lateral >= edges[index]) & (lateral < edges[index + 1])
        if np.any(selected):
            section[index] = float(np.percentile(anterior[selected], 90.0))
    available = np.flatnonzero(np.isfinite(section))
    if len(available) < max(9, bin_count // 5):
        raise RuntimeError(f"Insufficient transverse bridge coverage at {level_mm:.3f} mm")
    section = np.interp(np.arange(bin_count), available, section[available])
    return centers, _smooth_series(section, sigma_bins=1.0)


def _section_width(lateral_mm: np.ndarray, anterior_mm: np.ndarray, depth_mm: float) -> float:
    ridge = float(np.max(anterior_mm))
    within = anterior_mm >= ridge - depth_mm
    if np.count_nonzero(within) < 2:
        return 0.0
    return float(np.max(lateral_mm[within]) - np.min(lateral_mm[within]))


def _cross_section_diagnostics(
    source_vertices_m: np.ndarray,
    target_vertices_m: np.ndarray,
    simulated_vertices_m: np.ndarray,
    frame: dict[str, Any],
    apex_normalized: float,
    half_width_mm: float,
    profile_core_half_width_mm: float,
    bridge_core_half_width_mm: float,
) -> dict[str, Any]:
    source_longitudinal, source_lateral, source_anterior = _coordinates(
        source_vertices_m * 1000.0, frame
    )
    target_longitudinal, target_lateral, target_anterior = _coordinates(
        target_vertices_m * 1000.0, frame
    )
    simulated_longitudinal, simulated_lateral, simulated_anterior = _coordinates(
        simulated_vertices_m * 1000.0, frame
    )
    start = float(frame["roi_start_mm"])
    span = float(frame["roi_end_mm"]) - start
    levels = {
        "upper_dorsum": 0.18,
        "hump_region": apex_normalized,
        "mid_dorsum": 0.64,
        "supratip": 0.88,
    }
    longitudinal_window_mm = max(0.8, 0.025 * span)
    sections: dict[str, Any] = {}
    for name, normalized in levels.items():
        level_mm = start + normalized * span
        source_x, source_y = _sample_cross_section(
            source_longitudinal,
            source_lateral,
            source_anterior,
            level_mm=level_mm,
            half_width_mm=half_width_mm,
            longitudinal_window_mm=longitudinal_window_mm,
        )
        target_x, target_y = _sample_cross_section(
            target_longitudinal,
            target_lateral,
            target_anterior,
            level_mm=level_mm,
            half_width_mm=half_width_mm,
            longitudinal_window_mm=longitudinal_window_mm,
        )
        simulated_x, simulated_y = _sample_cross_section(
            simulated_longitudinal,
            simulated_lateral,
            simulated_anterior,
            level_mm=level_mm,
            half_width_mm=half_width_mm,
            longitudinal_window_mm=longitudinal_window_mm,
        )
        central = np.abs(simulated_x) <= min(4.5, 0.35 * half_width_mm)
        simulated_curvature = float(np.polyfit(simulated_x[central], simulated_y[central], 2)[0])
        source_curvature = float(np.polyfit(source_x[central], source_y[central], 2)[0])
        source_envelope_width_mm = _section_width(source_x, source_y, 1.5)
        target_envelope_width_mm = _section_width(target_x, target_y, 1.5)
        simulated_envelope_width_mm = _section_width(simulated_x, simulated_y, 1.5)

        # Measure the two actual sidewalls by vertex identity.  Reporting the
        # sampled envelope width alone can hide a lateral deformation when a
        # neighboring bin becomes the new 1.5 mm threshold crossing.
        near_level = np.abs(source_longitudinal - level_mm) <= longitudinal_window_mm
        source_level_anterior = source_anterior[near_level]
        ridge_anterior_mm = (
            float(np.percentile(source_level_anterior, 98.0))
            if len(source_level_anterior)
            else float("-inf")
        )
        surface_band = source_anterior >= ridge_anterior_mm - 5.0
        matched_bridge = near_level & (source_anterior >= ridge_anterior_mm - 1.5)
        width_before_mm = (
            float(np.ptp(source_lateral[matched_bridge]))
            if np.count_nonzero(matched_bridge) >= 2
            else source_envelope_width_mm
        )
        width_target_mm = (
            float(np.ptp(target_lateral[matched_bridge]))
            if np.count_nonzero(matched_bridge) >= 2
            else target_envelope_width_mm
        )
        width_after_mm = (
            float(np.ptp(simulated_lateral[matched_bridge]))
            if np.count_nonzero(matched_bridge) >= 2
            else simulated_envelope_width_mm
        )
        sidewall_inner_mm = 0.4 * bridge_core_half_width_mm
        sidewall_outer_mm = min(1.15 * bridge_core_half_width_mm, 0.8 * half_width_mm)
        lateral_change_mm = simulated_lateral - source_lateral
        left_sidewall = (
            near_level
            & surface_band
            & (source_lateral <= -sidewall_inner_mm)
            & (source_lateral >= -sidewall_outer_mm)
        )
        right_sidewall = (
            near_level
            & surface_band
            & (source_lateral >= sidewall_inner_mm)
            & (source_lateral <= sidewall_outer_mm)
        )
        left_medial_mm = np.maximum(lateral_change_mm[left_sidewall], 0.0)
        right_medial_mm = np.maximum(-lateral_change_mm[right_sidewall], 0.0)
        left_sidewall_displacement_mm = (
            float(np.median(left_medial_mm[left_medial_mm > 1e-6]))
            if np.any(left_medial_mm > 1e-6)
            else 0.0
        )
        right_sidewall_displacement_mm = (
            float(np.median(right_medial_mm[right_medial_mm > 1e-6]))
            if np.any(right_medial_mm > 1e-6)
            else 0.0
        )
        central_height_before_mm = float(np.interp(0.0, source_x, source_y))
        central_height_target_mm = float(np.interp(0.0, target_x, target_y))
        central_height_after_mm = float(np.interp(0.0, simulated_x, simulated_y))

        def sidewall_position(mask: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
            if not np.any(mask):
                return {"lateral_mm": 0.0, "anterior_mm": 0.0}
            return {
                "lateral_mm": round(float(np.median(x[mask])), 6),
                "anterior_mm": round(float(np.median(y[mask])), 6),
            }

        left_sidewall_before = sidewall_position(left_sidewall, source_lateral, source_anterior)
        left_sidewall_target = sidewall_position(left_sidewall, target_lateral, target_anterior)
        left_sidewall_after = sidewall_position(
            left_sidewall, simulated_lateral, simulated_anterior
        )
        right_sidewall_before = sidewall_position(right_sidewall, source_lateral, source_anterior)
        right_sidewall_target = sidewall_position(
            right_sidewall, target_lateral, target_anterior
        )
        right_sidewall_after = sidewall_position(
            right_sidewall, simulated_lateral, simulated_anterior
        )
        ridge_core = np.abs(simulated_x) <= profile_core_half_width_mm
        ridge_shoulders = (np.abs(simulated_x) > profile_core_half_width_mm) & (
            np.abs(simulated_x) <= bridge_core_half_width_mm
        )
        central_ridge_prominence_mm = (
            float(central_height_after_mm - np.max(simulated_y[ridge_shoulders]))
            if np.any(ridge_core) and np.any(ridge_shoulders)
            else 0.0
        )
        sections[name] = {
            "normalized_nasion_to_supratip": round(float(normalized), 6),
            "longitudinal_mm_from_nasion": round(level_mm, 6),
            "width_before_mm": round(width_before_mm, 6),
            "width_target_mm": round(width_target_mm, 6),
            "width_after_mm": round(width_after_mm, 6),
            "central_height_before_mm": round(central_height_before_mm, 6),
            "central_height_target_mm": round(central_height_target_mm, 6),
            "central_height_after_mm": round(central_height_after_mm, 6),
            "central_posterior_displacement_mm": round(
                central_height_before_mm - central_height_after_mm, 6
            ),
            "left_sidewall_position_before": left_sidewall_before,
            "left_sidewall_position_target": left_sidewall_target,
            "left_sidewall_position_after": left_sidewall_after,
            "right_sidewall_position_before": right_sidewall_before,
            "right_sidewall_position_target": right_sidewall_target,
            "right_sidewall_position_after": right_sidewall_after,
            "left_sidewall_displacement_mm": round(left_sidewall_displacement_mm, 6),
            "right_sidewall_displacement_mm": round(right_sidewall_displacement_mm, 6),
            "left_sidewall_sample_count": int(np.count_nonzero(left_sidewall)),
            "right_sidewall_sample_count": int(np.count_nonzero(right_sidewall)),
            "source_lateral_mm": source_x.round(6).tolist(),
            "source_anterior_mm": source_y.round(6).tolist(),
            "target_lateral_mm": target_x.round(6).tolist(),
            "target_anterior_mm": target_y.round(6).tolist(),
            "simulated_lateral_mm": simulated_x.round(6).tolist(),
            "simulated_anterior_mm": simulated_y.round(6).tolist(),
            "source_width_at_1_5mm_depth_mm": round(source_envelope_width_mm, 6),
            "target_width_at_1_5mm_depth_mm": round(target_envelope_width_mm, 6),
            "simulated_width_at_1_5mm_depth_mm": round(simulated_envelope_width_mm, 6),
            "source_ridge_lateral_mm": round(float(source_x[np.argmax(source_y)]), 6),
            "simulated_ridge_lateral_mm": round(float(simulated_x[np.argmax(simulated_y)]), 6),
            "source_central_quadratic_curvature_per_mm": round(source_curvature, 8),
            "simulated_central_quadratic_curvature_per_mm": round(simulated_curvature, 8),
            "final_central_ridge_prominence_mm": round(central_ridge_prominence_mm, 6),
            "final_has_central_dent": central_ridge_prominence_mm < -0.05,
        }
    return {
        "definition": "frontal transverse anterior envelope",
        "width_definition": (
            "lateral span of source-identical bridge vertices selected within 1.5 mm posterior "
            "to the source dorsal ridge"
        ),
        "envelope_width_definition": (
            "independent source/simulation lateral envelope within 1.5 mm of each dorsal ridge"
        ),
        "sidewall_displacement_definition": (
            "median medial displacement of source-identical dorsal sidewall vertices"
        ),
        "sidewall_position_definition": (
            "median lateral/anterior coordinates of source-identical dorsal sidewall vertices"
        ),
        "central_height_definition": (
            "anterior coordinate of the sampled transverse envelope at the patient midline"
        ),
        "deformation_model": "one coupled vector solve over the connected dorsal vault",
        "longitudinal_sampling_half_window_mm": round(longitudinal_window_mm, 6),
        "sections": sections,
    }


def _acceptance_diagnostics(
    profile: dict[str, np.ndarray],
    cross_sections: dict[str, Any],
    frame: dict[str, Any],
    applied_reduction_mm: float,
    vault_coverage: dict[str, Any],
) -> dict[str, Any]:
    """Summarize whether the numerical result meets the profile/vault objectives."""
    applied_reduction_mm = float(applied_reduction_mm)
    if applied_reduction_mm <= 0.0:
        return {
            "passed": True,
            "identity_request": True,
            "profile": {"passed": True, "active_upper_mid_sample_count": 0},
            "frontal_vault": {
                "passed": True,
                "corrected_section_count": 0,
                "no_center_strip_pattern": True,
            },
        }

    longitudinal = np.asarray(profile["longitudinal_mm"], dtype=np.float64)
    requested = np.asarray(profile["centerline_reduction_mm"], dtype=np.float64)
    achieved = np.asarray(profile["source_anterior_mm"], dtype=np.float64) - np.asarray(
        profile["final_anterior_mm"], dtype=np.float64
    )
    target_error = np.asarray(profile["final_anterior_mm"], dtype=np.float64) - np.asarray(
        profile["target_anterior_mm"], dtype=np.float64
    )
    normalized = (longitudinal - float(frame["roi_start_mm"])) / (
        float(frame["roi_end_mm"]) - float(frame["roi_start_mm"])
    )
    active = (
        (normalized >= 0.08)
        & (normalized <= 0.72)
        & (requested >= max(0.05, 0.05 * applied_reduction_mm))
    )
    lower_band = (normalized >= 0.70) & (normalized <= 0.82)
    maximum_lower_requested_mm = (
        float(np.max(requested[lower_band])) if np.any(lower_band) else 0.0
    )
    lower_to_peak_ratio = maximum_lower_requested_mm / max(applied_reduction_mm, 1e-9)
    lower_dorsum_not_overcorrected = (
        lower_to_peak_ratio <= MAX_LOWER_DORSUM_CORRECTION_FRACTION
    )
    if np.any(active):
        achievement_ratio = achieved[active] / np.maximum(requested[active], 1e-9)
        p10_achievement_ratio = float(np.percentile(achievement_ratio, 10.0))
        maximum_target_error_mm = float(np.max(np.abs(target_error[active])))
        profile_passed = (
            p10_achievement_ratio >= 0.85
            and maximum_target_error_mm <= max(0.25, 0.15 * applied_reduction_mm)
            and lower_dorsum_not_overcorrected
        )
    else:
        p10_achievement_ratio = 0.0
        maximum_target_error_mm = float("inf")
        profile_passed = False

    corrected_sections = [
        section
        for section in cross_sections["sections"].values()
        if section["central_posterior_displacement_mm"]
        >= max(0.1, 0.05 * applied_reduction_mm)
    ]
    no_central_dent = bool(
        corrected_sections and all(not section["final_has_central_dent"] for section in corrected_sections)
    )
    no_width_expansion = bool(
        corrected_sections
        and all(
            section["width_after_mm"] <= section["width_before_mm"] + 0.05
            for section in corrected_sections
        )
    )
    sidewalls_not_outward = bool(
        corrected_sections
        and all(
            section["left_sidewall_position_after"]["lateral_mm"]
            >= section["left_sidewall_position_before"]["lateral_mm"] - 0.02
            and section["right_sidewall_position_after"]["lateral_mm"]
            <= section["right_sidewall_position_before"]["lateral_mm"] + 0.02
            for section in corrected_sections
        )
    )
    mild_narrowing = bool(
        corrected_sections
        and all(
            0.0
            <= section["width_before_mm"] - section["width_after_mm"]
            <= 2.0 * MAX_TRANSVERSE_MEDIAL_ADJUSTMENT_MM + 0.05
            for section in corrected_sections
        )
    )
    no_center_strip_pattern = not bool(vault_coverage["center_strip_pattern_detected"])
    frontal_passed = (
        no_central_dent
        and no_width_expansion
        and sidewalls_not_outward
        and mild_narrowing
        and no_center_strip_pattern
    )
    return {
        "passed": bool(profile_passed and frontal_passed),
        "identity_request": False,
        "profile": {
            "passed": bool(profile_passed),
            "active_upper_mid_sample_count": int(np.count_nonzero(active)),
            "p10_achieved_to_requested_correction_ratio": round(p10_achievement_ratio, 6),
            "maximum_absolute_target_error_mm": round(maximum_target_error_mm, 6),
            "superior_hump_not_left_behind": p10_achievement_ratio >= 0.85,
            "smooth_target_followed": maximum_target_error_mm
            <= max(0.25, 0.15 * applied_reduction_mm),
            "lower_dorsum_not_overcorrected": lower_dorsum_not_overcorrected,
            "maximum_lower_dorsum_requested_correction_mm": round(
                maximum_lower_requested_mm, 6
            ),
            "lower_to_peak_requested_correction_ratio": round(lower_to_peak_ratio, 6),
        },
        "frontal_vault": {
            "passed": bool(frontal_passed),
            "corrected_section_count": len(corrected_sections),
            "no_central_dent": no_central_dent,
            "no_width_expansion": no_width_expansion,
            "sidewalls_not_moved_outward": sidewalls_not_outward,
            "narrowing_within_mild_limit": mild_narrowing,
            "no_center_strip_pattern": no_center_strip_pattern,
            "slope_to_ridge_median_displacement_ratio": vault_coverage[
                "slope_to_ridge_median_ratio"
            ],
            "sidewall_to_ridge_median_displacement_ratio": vault_coverage[
                "sidewall_to_ridge_median_ratio"
            ],
        },
    }


def _svg_cross_section_points(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    offset_x: float,
    offset_y: float,
    x_min: float,
    y_min: float,
    x_span: float,
    y_span: float,
    panel_width: int,
    panel_height: int,
    margin: int,
) -> str:
    px = offset_x + margin + (x_values - x_min) / x_span * (panel_width - 2 * margin)
    py = (
        offset_y + panel_height - margin - (y_values - y_min) / y_span * (panel_height - 2 * margin)
    )
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(px, py))


def _write_cross_section_svg(output_path: Path, diagnostic: dict[str, Any]) -> None:
    panel_width = 520
    panel_height = 380
    margin = 55
    sections = list(diagnostic["sections"].items())
    svg_parts = [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="760" '
            'viewBox="0 0 1040 760">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for panel_index, (name, section) in enumerate(sections):
        column = panel_index % 2
        row = panel_index // 2
        offset_x = column * panel_width
        offset_y = row * panel_height
        source_x = np.asarray(section["source_lateral_mm"], dtype=np.float64)
        source_y = np.asarray(section["source_anterior_mm"], dtype=np.float64)
        target_x = np.asarray(section["target_lateral_mm"], dtype=np.float64)
        target_y = np.asarray(section["target_anterior_mm"], dtype=np.float64)
        simulated_x = np.asarray(section["simulated_lateral_mm"], dtype=np.float64)
        simulated_y = np.asarray(section["simulated_anterior_mm"], dtype=np.float64)
        all_x = np.concatenate([source_x, target_x, simulated_x])
        all_y = np.concatenate([source_y, target_y, simulated_y])
        x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
        y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
        x_span = max(x_max - x_min, 1e-9)
        y_span = max(y_max - y_min, 1e-9)

        source_points = _svg_cross_section_points(
            source_x,
            source_y,
            offset_x=offset_x,
            offset_y=offset_y,
            x_min=x_min,
            y_min=y_min,
            x_span=x_span,
            y_span=y_span,
            panel_width=panel_width,
            panel_height=panel_height,
            margin=margin,
        )
        simulated_points = _svg_cross_section_points(
            simulated_x,
            simulated_y,
            offset_x=offset_x,
            offset_y=offset_y,
            x_min=x_min,
            y_min=y_min,
            x_span=x_span,
            y_span=y_span,
            panel_width=panel_width,
            panel_height=panel_height,
            margin=margin,
        )
        target_points = _svg_cross_section_points(
            target_x,
            target_y,
            offset_x=offset_x,
            offset_y=offset_y,
            x_min=x_min,
            y_min=y_min,
            x_span=x_span,
            y_span=y_span,
            panel_width=panel_width,
            panel_height=panel_height,
            margin=margin,
        )

        title = name.replace("_", " ")
        source_width = section["source_width_at_1_5mm_depth_mm"]
        simulated_width = section["simulated_width_at_1_5mm_depth_mm"]
        svg_parts.extend(
            [
                (
                    f'<rect x="{offset_x + margin}" y="{offset_y + margin}" '
                    f'width="{panel_width - 2 * margin}" '
                    f'height="{panel_height - 2 * margin}" fill="none" stroke="#555"/>'
                ),
                (
                    f'<polyline points="{source_points}" fill="none" '
                    'stroke="#d62728" stroke-width="3"/>'
                ),
                (
                    f'<polyline points="{target_points}" fill="none" '
                    'stroke="#2ca02c" stroke-width="2" stroke-dasharray="6 4"/>'
                ),
                (
                    f'<polyline points="{simulated_points}" fill="none" '
                    'stroke="#1f77b4" stroke-width="3"/>'
                ),
                (
                    f'<text x="{offset_x + margin}" y="{offset_y + 28}" '
                    f'font-family="sans-serif" font-size="18">{title}</text>'
                ),
                (
                    f'<text x="{offset_x + margin}" y="{offset_y + 48}" '
                    f'font-family="sans-serif" font-size="12">'
                    f"red source {source_width:.2f} mm | green target | "
                    f"blue final {simulated_width:.2f} mm</text>"
                ),
            ]
        )
    svg_parts.append("</svg>")
    output_path.write_text("\n".join(svg_parts), encoding="utf-8")


def _write_profile_svg(output_path: Path, profile: dict[str, np.ndarray]) -> float:
    width = 900
    height = 520
    margin = 70
    x = profile["longitudinal_mm"]
    source = profile["source_anterior_mm"]
    target = profile["target_anterior_mm"]
    final = profile["final_anterior_mm"]
    y_values = np.concatenate([source, target, final])
    x_span = max(float(np.ptp(x)), 1e-9)
    y_min = float(np.min(y_values))
    y_span = max(float(np.ptp(y_values)), 1e-9)

    def points(values: np.ndarray) -> str:
        px = margin + (x - float(np.min(x))) / x_span * (width - 2 * margin)
        py = height - margin - (values - y_min) / y_span * (height - 2 * margin)
        return " ".join(f"{x_value:.2f},{y_value:.2f}" for x_value, y_value in zip(px, py))

    maximum_profile_change_mm = float(np.max(np.abs(source - final)))
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333"/>
<polyline points="{points(target)}" fill="none" stroke="#888" stroke-width="2" stroke-dasharray="7 5"/>
<polyline points="{points(source)}" fill="none" stroke="#d62728" stroke-width="4"/>
<polyline points="{points(final)}" fill="none" stroke="#1f77b4" stroke-width="3"/>
<text x="{margin}" y="32" font-family="sans-serif" font-size="20">Dorsal profile diagnostic</text>
<text x="{margin}" y="54" font-family="sans-serif" font-size="14">red: original | gray: target | blue: final | max change: {maximum_profile_change_mm:.3f} mm</text>
<text x="{width / 2 - 90}" y="{height - 18}" font-family="sans-serif" font-size="14">nasion to supratip (mm)</text>
<text x="18" y="{height / 2}" font-family="sans-serif" font-size="14" transform="rotate(-90 18 {height / 2})">posterior to anterior (mm)</text>
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")
    return maximum_profile_change_mm


def _projection_image(
    horizontal: np.ndarray,
    vertical: np.ndarray,
    depth: np.ndarray,
    *,
    title: str,
    horizontal_label: str,
    vertical_label: str,
    vertical_increases_down: bool,
    highlight_mask: np.ndarray | None = None,
    point_colors_bgr: np.ndarray | None = None,
    bounds: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    import cv2

    height = 900
    width = 900
    margin = 65
    if bounds is None:
        x_min, x_max = np.percentile(horizontal, [0.5, 99.5])
        y_min, y_max = np.percentile(vertical, [0.5, 99.5])
        x_padding = max(0.04 * float(x_max - x_min), 1e-6)
        y_padding = max(0.04 * float(y_max - y_min), 1e-6)
        bounds = (
            float(x_min - x_padding),
            float(x_max + x_padding),
            float(y_min - y_padding),
            float(y_max + y_padding),
        )
    x_min, x_max, y_min, y_max = bounds
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)
    px = np.rint(margin + (horizontal - x_min) / x_span * (width - 2 * margin)).astype(int)
    vertical_fraction = (vertical - y_min) / y_span
    if not vertical_increases_down:
        vertical_fraction = 1.0 - vertical_fraction
    py = np.rint(margin + vertical_fraction * (height - 2 * margin)).astype(int)
    valid = (px >= margin) & (px < width - margin) & (py >= margin) & (py < height - margin)

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    depth_min, depth_max = np.percentile(depth[valid], [2.0, 98.0])
    depth_span = max(float(depth_max - depth_min), 1e-9)
    shade = 195.0 - 130.0 * np.clip((depth - depth_min) / depth_span, 0.0, 1.0)
    order = np.argsort(depth[valid], kind="stable")
    valid_indices = np.flatnonzero(valid)[order]
    if point_colors_bgr is None:
        gray = shade[valid_indices].astype(np.uint8)
        rendered_colors = np.column_stack([gray, gray, gray])
    else:
        colors = np.asarray(point_colors_bgr, dtype=np.uint8)
        if colors.shape != (len(horizontal), 3):
            raise ValueError("Projection colors must contain one BGR triplet per vertex")
        rendered_colors = colors[valid_indices]
    canvas[py[valid_indices], px[valid_indices]] = rendered_colors
    canvas = cv2.erode(canvas, np.ones((2, 2), dtype=np.uint8), iterations=1)

    if highlight_mask is not None:
        highlighted = valid & highlight_mask
        overlay = np.zeros((height, width), dtype=np.uint8)
        overlay[py[highlighted], px[highlighted]] = 255
        overlay = cv2.dilate(overlay, np.ones((5, 5), dtype=np.uint8), iterations=1)
        canvas[overlay > 0] = (35, 35, 225)

    cv2.rectangle(canvas, (margin, margin), (width - margin, height - margin), (60, 60, 60), 1)
    cv2.putText(canvas, title, (margin, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2)
    cv2.putText(
        canvas,
        horizontal_label,
        (width // 2 - 120, height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        1,
    )
    cv2.putText(
        canvas,
        vertical_label,
        (8, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (20, 20, 20),
        1,
    )
    return canvas, bounds


def _write_diagnostic_renders(
    output_dir: Path,
    label: str,
    source_vertices_m: np.ndarray,
    simulated_vertices_m: np.ndarray,
    faces: np.ndarray,
    frame: dict[str, Any],
    displacement_mm: np.ndarray,
    source_vertex_colors: np.ndarray | None = None,
) -> dict[str, Path]:
    import cv2

    source_longitudinal, source_lateral, source_anterior = _coordinates(
        source_vertices_m * 1000.0,
        frame,
    )
    simulated_longitudinal, simulated_lateral, simulated_anterior = _coordinates(
        simulated_vertices_m * 1000.0,
        frame,
    )
    source_normals = _vertex_normals(source_vertices_m, faces)
    simulated_normals = _vertex_normals(simulated_vertices_m, faces)
    displacement_mm = np.asarray(displacement_mm, dtype=np.float64)
    texture_colors_bgr: np.ndarray | None = None
    if source_vertex_colors is not None and len(source_vertex_colors) == len(source_vertices_m):
        texture_rgb = np.asarray(source_vertex_colors, dtype=np.float64)
        if float(np.max(texture_rgb)) <= 1.0:
            texture_rgb = texture_rgb * 255.0
        texture_colors_bgr = np.clip(texture_rgb[:, :3], 0.0, 255.0).astype(np.uint8)[:, ::-1]
    affected_vertex_mask = displacement_mm > 0.1
    anterior_axis = np.asarray(frame["anterior"])
    if float(np.median(source_normals @ anterior_axis)) < 0.0:
        source_normals *= -1.0
    if float(np.median(simulated_normals @ anterior_axis)) < 0.0:
        simulated_normals *= -1.0
    light_direction = _unit(
        0.25 * np.asarray(frame["lateral"]) - 0.2 * np.asarray(frame["vertical"]) + anterior_axis,
        "diagnostic light direction",
    )

    def clay_colors(normals: np.ndarray) -> np.ndarray:
        illumination = 0.3 + 0.7 * np.clip(normals @ light_direction, 0.0, 1.0)
        gray = np.clip(45.0 + 190.0 * illumination, 0.0, 255.0).astype(np.uint8)
        return np.column_stack([gray, gray, gray])

    def normal_colors(normals: np.ndarray) -> np.ndarray:
        anatomical = np.column_stack(
            [
                normals @ np.asarray(frame["lateral"]),
                normals @ np.asarray(frame["vertical"]),
                normals @ anterior_axis,
            ]
        )
        rgb = np.clip((anatomical + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
        return rgb[:, ::-1]

    front_before, front_bounds = _projection_image(
        source_lateral,
        source_longitudinal,
        source_anterior,
        title="Textured front view - source",
        horizontal_label="patient left to right",
        vertical_label="superior to inferior",
        vertical_increases_down=True,
        point_colors_bgr=texture_colors_bgr,
    )
    front_after, _ = _projection_image(
        simulated_lateral,
        simulated_longitudinal,
        simulated_anterior,
        title="Textured front view - final",
        horizontal_label="patient left to right",
        vertical_label="superior to inferior",
        vertical_increases_down=True,
        point_colors_bgr=texture_colors_bgr,
        bounds=front_bounds,
    )
    profile_before, profile_bounds = _projection_image(
        source_longitudinal,
        source_anterior,
        -np.abs(source_lateral),
        title="Textured profile view - source",
        horizontal_label="nasion to inferior",
        vertical_label="posterior to anterior",
        vertical_increases_down=False,
        point_colors_bgr=texture_colors_bgr,
    )
    profile_after, _ = _projection_image(
        simulated_longitudinal,
        simulated_anterior,
        -np.abs(simulated_lateral),
        title="Textured profile view - final",
        horizontal_label="nasion to inferior",
        vertical_label="posterior to anterior",
        vertical_increases_down=False,
        point_colors_bgr=texture_colors_bgr,
        bounds=profile_bounds,
    )
    clay_before, _ = _projection_image(
        source_lateral,
        source_longitudinal,
        source_anterior,
        title="Clay front view - source",
        horizontal_label="patient left to right",
        vertical_label="superior to inferior",
        vertical_increases_down=True,
        point_colors_bgr=clay_colors(source_normals),
        bounds=front_bounds,
    )
    clay_after, _ = _projection_image(
        simulated_lateral,
        simulated_longitudinal,
        simulated_anterior,
        title="Clay front view - simulated",
        horizontal_label="patient left to right",
        vertical_label="superior to inferior",
        vertical_increases_down=True,
        point_colors_bgr=clay_colors(simulated_normals),
        bounds=front_bounds,
    )
    profile_clay_before, _ = _projection_image(
        source_longitudinal,
        source_anterior,
        -np.abs(source_lateral),
        title="Clay profile view - source",
        horizontal_label="nasion to inferior",
        vertical_label="posterior to anterior",
        vertical_increases_down=False,
        point_colors_bgr=clay_colors(source_normals),
        bounds=profile_bounds,
    )
    profile_clay_after, _ = _projection_image(
        simulated_longitudinal,
        simulated_anterior,
        -np.abs(simulated_lateral),
        title="Clay profile view - simulated",
        horizontal_label="nasion to inferior",
        vertical_label="posterior to anterior",
        vertical_increases_down=False,
        point_colors_bgr=clay_colors(simulated_normals),
        bounds=profile_bounds,
    )
    top_down_before, top_down_bounds = _projection_image(
        source_lateral,
        source_anterior,
        -source_longitudinal,
        title="Clay top-down view - source",
        horizontal_label="patient left to right",
        vertical_label="posterior to anterior",
        vertical_increases_down=False,
        point_colors_bgr=clay_colors(source_normals),
    )
    top_down_after, _ = _projection_image(
        simulated_lateral,
        simulated_anterior,
        -simulated_longitudinal,
        title="Clay top-down view - simulated",
        horizontal_label="patient left to right",
        vertical_label="posterior to anterior",
        vertical_increases_down=False,
        point_colors_bgr=clay_colors(simulated_normals),
        bounds=top_down_bounds,
    )
    normal_before, _ = _projection_image(
        source_lateral,
        source_longitudinal,
        source_anterior,
        title="Normal visualization - source",
        horizontal_label="patient left to right",
        vertical_label="superior to inferior",
        vertical_increases_down=True,
        point_colors_bgr=normal_colors(source_normals),
        bounds=front_bounds,
    )
    normal_after, _ = _projection_image(
        simulated_lateral,
        simulated_longitudinal,
        simulated_anterior,
        title="Normal visualization - simulated",
        horizontal_label="patient left to right",
        vertical_label="superior to inferior",
        vertical_increases_down=True,
        point_colors_bgr=normal_colors(simulated_normals),
        bounds=front_bounds,
    )
    highlighted_front, _ = _projection_image(
        source_lateral,
        source_longitudinal,
        source_anterior,
        title="Affected dorsal ROI - front/profile",
        horizontal_label="patient left to right",
        vertical_label="superior to inferior",
        vertical_increases_down=True,
        highlight_mask=affected_vertex_mask,
        bounds=front_bounds,
    )
    highlighted_profile, _ = _projection_image(
        source_longitudinal,
        source_anterior,
        -np.abs(source_lateral),
        title="Affected dorsal ROI - profile",
        horizontal_label="nasion to inferior",
        vertical_label="posterior to anterior",
        vertical_increases_down=False,
        highlight_mask=affected_vertex_mask,
        bounds=profile_bounds,
    )
    roi_highlight = np.concatenate([highlighted_front, highlighted_profile], axis=1)
    heat_scale = max(float(np.max(displacement_mm)), 1e-9)
    heat_values = np.rint(255.0 * np.clip(displacement_mm / heat_scale, 0.0, 1.0)).astype(np.uint8)
    heat_colors = cv2.applyColorMap(heat_values[:, None], cv2.COLORMAP_TURBO).reshape(-1, 3)
    heat_colors[displacement_mm <= 0.1] = (220, 220, 220)
    heatmap_front, _ = _projection_image(
        simulated_lateral,
        simulated_longitudinal,
        simulated_anterior,
        title=f"Moved vertices - front (max {heat_scale:.2f} mm)",
        horizontal_label="patient left to right",
        vertical_label="superior to inferior",
        vertical_increases_down=True,
        point_colors_bgr=heat_colors,
        bounds=front_bounds,
    )
    heatmap_profile, _ = _projection_image(
        simulated_longitudinal,
        simulated_anterior,
        -np.abs(simulated_lateral),
        title=f"Moved vertices - profile (max {heat_scale:.2f} mm)",
        horizontal_label="nasion to inferior",
        vertical_label="posterior to anterior",
        vertical_increases_down=False,
        point_colors_bgr=heat_colors,
        bounds=profile_bounds,
    )
    moved_vertices_heatmap = np.concatenate([heatmap_front, heatmap_profile], axis=1)
    paths = {
        "front_before_png": output_dir / f"reduction_{label}mm_front_before.png",
        "front_after_png": output_dir / f"reduction_{label}mm_front_after.png",
        "profile_before_png": output_dir / f"reduction_{label}mm_profile_before.png",
        "profile_after_png": output_dir / f"reduction_{label}mm_profile_after.png",
        "textured_front_before_png": (
            output_dir / f"reduction_{label}mm_textured_front_before.png"
        ),
        "textured_front_after_png": output_dir / f"reduction_{label}mm_textured_front_after.png",
        "textured_profile_before_png": (
            output_dir / f"reduction_{label}mm_textured_profile_before.png"
        ),
        "textured_profile_after_png": (
            output_dir / f"reduction_{label}mm_textured_profile_after.png"
        ),
        "affected_roi_render_png": output_dir / f"reduction_{label}mm_affected_roi.png",
        "clay_before_png": output_dir / f"reduction_{label}mm_clay_before.png",
        "clay_after_png": output_dir / f"reduction_{label}mm_clay_after.png",
        "profile_clay_before_png": output_dir / f"reduction_{label}mm_profile_clay_before.png",
        "profile_clay_after_png": output_dir / f"reduction_{label}mm_profile_clay_after.png",
        "top_down_before_png": output_dir / f"reduction_{label}mm_top_down_before.png",
        "top_down_after_png": output_dir / f"reduction_{label}mm_top_down_after.png",
        "moved_vertices_heatmap_png": (
            output_dir / f"reduction_{label}mm_moved_vertices_heatmap.png"
        ),
        "front_displacement_heatmap_png": (
            output_dir / f"reduction_{label}mm_front_displacement_heatmap.png"
        ),
        "profile_displacement_heatmap_png": (
            output_dir / f"reduction_{label}mm_profile_displacement_heatmap.png"
        ),
        "normal_before_png": output_dir / f"reduction_{label}mm_normals_before.png",
        "normal_after_png": output_dir / f"reduction_{label}mm_normals_after.png",
    }
    images = {
        "front_before_png": front_before,
        "front_after_png": front_after,
        "profile_before_png": profile_before,
        "profile_after_png": profile_after,
        "textured_front_before_png": front_before,
        "textured_front_after_png": front_after,
        "textured_profile_before_png": profile_before,
        "textured_profile_after_png": profile_after,
        "affected_roi_render_png": roi_highlight,
        "clay_before_png": clay_before,
        "clay_after_png": clay_after,
        "profile_clay_before_png": profile_clay_before,
        "profile_clay_after_png": profile_clay_after,
        "top_down_before_png": top_down_before,
        "top_down_after_png": top_down_after,
        "moved_vertices_heatmap_png": moved_vertices_heatmap,
        "front_displacement_heatmap_png": heatmap_front,
        "profile_displacement_heatmap_png": heatmap_profile,
        "normal_before_png": normal_before,
        "normal_after_png": normal_after,
    }
    for name, path in paths.items():
        if not cv2.imwrite(str(path), images[name]):
            raise RuntimeError(f"Failed to write simulation diagnostic render: {path}")
    return paths


def _export_simulation_glb(
    source_glb_path: Path | None,
    output_glb_path: Path,
    source_vertices_m: np.ndarray,
    simulated_vertices_m: np.ndarray,
    faces: np.ndarray,
    source_vertex_colors: np.ndarray | None,
    metadata: dict[str, Any],
    frame: dict[str, Any],
    bridge_core_half_width_mm: float,
) -> dict[str, Any]:
    import trimesh

    simulated_normals = _vertex_normals(simulated_vertices_m, faces)
    if source_glb_path is not None and source_glb_path.is_file():
        mesh = _single_mesh(source_glb_path).copy()
        render_faces = np.asarray(mesh.faces, dtype=np.int64)
        if render_faces.shape != faces.shape:
            raise RuntimeError("Source GLB does not retain authoritative triangle correspondence")
        source_corner_error = float(
            np.max(
                np.linalg.norm(
                    np.asarray(mesh.vertices)[render_faces] - source_vertices_m[faces],
                    axis=2,
                )
            )
        )
        if source_corner_error > 1e-6:
            raise RuntimeError("Source GLB is not registered to the authoritative geometry")
        render_indices = render_faces.reshape(-1)
        canonical_indices = faces.reshape(-1)
        order = np.argsort(render_indices, kind="stable")
        sorted_render = render_indices[order]
        sorted_canonical = canonical_indices[order]
        conflict = (sorted_render[1:] == sorted_render[:-1]) & (
            sorted_canonical[1:] != sorted_canonical[:-1]
        )
        if np.any(conflict):
            raise RuntimeError("Source GLB maps one render vertex to multiple source vertices")
        canonical_for_render = np.full(len(mesh.vertices), -1, dtype=np.int64)
        canonical_for_render[sorted_render] = sorted_canonical
        if np.any(canonical_for_render < 0):
            raise RuntimeError("Source GLB contains vertices outside the authoritative surface")
        mesh.vertices = simulated_vertices_m[canonical_for_render]
    else:
        vertex_colors = None
        if source_vertex_colors is not None:
            rgb = np.clip(source_vertex_colors * 255.0, 0, 255).astype(np.uint8)
            vertex_colors = np.column_stack([rgb, np.full(len(rgb), 255, dtype=np.uint8)])
        mesh = trimesh.Trimesh(
            vertices=simulated_vertices_m,
            faces=faces,
            vertex_colors=vertex_colors,
            process=False,
        )
        canonical_for_render = np.arange(len(simulated_vertices_m), dtype=np.int64)
    # Do not let normals inherited from the source GLB survive a geometry
    # change. Assign normals recomputed from the simulated vertex positions.
    mesh.vertex_normals = simulated_normals[canonical_for_render]
    scene = trimesh.Scene(mesh)

    def add_simulation_metadata(tree: dict) -> None:
        extras = {
            "schema_version": 1,
            "role": "non_authoritative_aesthetic_simulation",
            "operation": "dorsal_hump_reduction",
            "source_geometry_id": metadata["source_geometry_id"],
            "simulation_geometry_id": metadata["simulation_geometry_id"],
            "requested_reduction_mm": metadata["requested_reduction_mm"],
            "units": "metres",
            "clinical_prediction": False,
        }
        tree.setdefault("asset", {}).setdefault("extras", {})["rhino_poc"] = extras
        for gltf_mesh in tree.get("meshes", []):
            gltf_mesh["extras"] = extras
            for primitive in gltf_mesh.get("primitives", []):
                primitive["extras"] = extras

    output_glb_path.parent.mkdir(parents=True, exist_ok=True)
    output_glb_path.write_bytes(
        trimesh.exchange.gltf.export_glb(scene, tree_postprocessor=add_simulation_metadata)
    )
    persisted = _single_mesh(output_glb_path)
    persisted_faces = np.asarray(persisted.faces, dtype=np.int64)
    if persisted_faces.shape != faces.shape:
        raise RuntimeError("Simulation GLB changed the source triangle count")
    maximum_corner_error = float(
        np.max(
            np.linalg.norm(
                np.asarray(persisted.vertices)[persisted_faces] - simulated_vertices_m[faces],
                axis=2,
            )
        )
    )
    if maximum_corner_error > 1e-6:
        raise RuntimeError(
            f"Simulation GLB moved the requested surface by {maximum_corner_error:.3g} m"
        )
    persisted_corner_vertices_m = np.asarray(persisted.vertices)[persisted_faces]
    source_corner_vertices_m = source_vertices_m[faces]
    glb_corner_delta_mm = (persisted_corner_vertices_m - source_corner_vertices_m) * 1000.0
    glb_displacement_mm = np.linalg.norm(glb_corner_delta_mm, axis=2)
    transverse_corner_change_mm = glb_corner_delta_mm @ np.asarray(frame["lateral"])
    canonical_transverse_change_mm = np.zeros(len(source_vertices_m), dtype=np.float64)
    canonical_transverse_change_mm[faces.reshape(-1)] = transverse_corner_change_mm.reshape(-1)
    source_longitudinal, source_lateral, _ = _coordinates(source_vertices_m * 1000.0, frame)
    sidewall_inner_mm = 0.4 * bridge_core_half_width_mm
    sidewall_outer_mm = 1.15 * bridge_core_half_width_mm
    dorsal_span = (source_longitudinal > float(frame["roi_start_mm"])) & (
        source_longitudinal < float(frame["roi_end_mm"])
    )
    left_sidewall = (
        dorsal_span
        & (source_lateral <= -sidewall_inner_mm)
        & (source_lateral >= -sidewall_outer_mm)
    )
    right_sidewall = (
        dorsal_span & (source_lateral >= sidewall_inner_mm) & (source_lateral <= sidewall_outer_mm)
    )
    left_medial = canonical_transverse_change_mm[left_sidewall]
    right_medial = -canonical_transverse_change_mm[right_sidewall]
    left_medial = left_medial[left_medial > 0.1]
    right_medial = right_medial[right_medial > 0.1]
    persisted_normals = np.asarray(persisted.vertex_normals, dtype=np.float64)
    expected_corner_normals = simulated_normals[faces]
    persisted_corner_normals = persisted_normals[persisted_faces]
    normal_dot = np.sum(expected_corner_normals * persisted_corner_normals, axis=2)
    normal_error_degrees = np.degrees(np.arccos(np.clip(normal_dot, -1.0, 1.0)))
    maximum_normal_error_degrees = float(normal_error_degrees.max())
    return {
        "maximum_vertex_error_from_ply_mm": maximum_corner_error * 1000.0,
        "maximum_displacement_from_source_mm": float(np.max(glb_displacement_mm)),
        "maximum_transverse_displacement_from_source_mm": float(
            np.max(np.abs(canonical_transverse_change_mm))
        ),
        "vertices_with_transverse_change_over_0_1_mm": int(
            np.count_nonzero(np.abs(canonical_transverse_change_mm) > 0.1)
        ),
        "left_sidewall_median_medial_displacement_mm": (
            float(np.median(left_medial)) if len(left_medial) else 0.0
        ),
        "right_sidewall_median_medial_displacement_mm": (
            float(np.median(right_medial)) if len(right_medial) else 0.0
        ),
        "geometry_differs_from_source": bool(np.max(glb_displacement_mm) > 1e-5),
        "normals_recomputed_from_simulated_geometry": True,
        "maximum_normal_error_degrees": maximum_normal_error_degrees,
        "p95_normal_error_degrees": float(np.percentile(normal_error_degrees, 95)),
    }


def _reduction_label(reduction_mm: float) -> str:
    label = f"{reduction_mm:.3f}".rstrip("0").rstrip(".")
    return label if "." in label else f"{label}.0"


def simulate_dorsal_hump_reduction(
    authoritative_mesh_path: Path,
    geometry_metadata_path: Path,
    landmarks_path: Path,
    output_dir: Path,
    *,
    reduction_mm: float = 0.0,
    source_glb_path: Path | None = None,
) -> dict[str, Any]:
    """Persist one separate dorsal-hump simulation and its provenance manifest."""
    import open3d as o3d

    source_paths = [authoritative_mesh_path, geometry_metadata_path, landmarks_path]
    if source_glb_path is not None and source_glb_path.is_file():
        source_paths.append(source_glb_path)
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required simulation source is missing: {path}")
    source_hashes_before = {str(path.resolve()): _sha256_file(path) for path in source_paths}

    source = o3d.io.read_triangle_mesh(str(authoritative_mesh_path))
    vertices_m = np.asarray(source.vertices, dtype=np.float64)
    faces = np.asarray(source.triangles, dtype=np.int64)
    if not len(vertices_m) or not len(faces):
        raise RuntimeError(f"Authoritative mesh is empty: {authoritative_mesh_path}")
    geometry = json.loads(geometry_metadata_path.read_text(encoding="utf-8"))
    landmarks_document = json.loads(landmarks_path.read_text(encoding="utf-8"))
    source_identity = geometry_identity(vertices_m, faces)
    if source_identity["geometry_id"] != geometry.get("geometry_id"):
        raise RuntimeError("Authoritative mesh does not match geometry.json")
    if landmarks_document.get("geometry_id") != geometry.get("geometry_id"):
        raise RuntimeError("landmarks.json does not reference the authoritative geometry")
    landmarks_mm = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in landmarks_document.get("landmarks", {}).items()
    }
    (
        simulated_vertices_m,
        displacement_mm,
        roi_metadata,
        roi_vertex_mask,
        profile_diagnostic,
        target_vertices_m,
    ) = _compute_dorsal_hump_deformation(vertices_m, landmarks_mm, reduction_mm, faces)
    if not roi_metadata["candidate_vertex_count"]:
        raise RuntimeError("The landmark-derived nasal dorsum ROI contains no vertices")
    applied_reduction_mm = float(roi_metadata["profile_model"]["applied_peak_reduction_mm"])

    output_dir.mkdir(parents=True, exist_ok=True)
    label = _reduction_label(float(reduction_mm))
    output_ply = output_dir / f"reduction_{label}mm.ply"
    output_glb = output_dir / f"reduction_{label}mm.glb"
    output_roi_ply = output_dir / f"reduction_{label}mm_affected_roi.ply"
    output_moved_vertices_ply = output_dir / f"reduction_{label}mm_moved_vertices.ply"
    output_profile_svg = output_dir / f"reduction_{label}mm_profile.svg"
    output_profile_json = output_dir / f"reduction_{label}mm_profile.json"
    output_cross_sections_svg = output_dir / f"reduction_{label}mm_cross_sections.svg"
    output_cross_sections_json = output_dir / f"reduction_{label}mm_cross_sections.json"
    output_manifest = output_dir / "simulation.json"
    if float(reduction_mm) == 0.0:
        shutil.copy2(authoritative_mesh_path, output_ply)
    else:
        simulated_mesh = o3d.geometry.TriangleMesh(source)
        simulated_mesh.vertices = o3d.utility.Vector3dVector(simulated_vertices_m)
        simulated_mesh.compute_vertex_normals()
        if not o3d.io.write_triangle_mesh(str(output_ply), simulated_mesh, write_ascii=False):
            raise RuntimeError(f"Failed to write simulated mesh: {output_ply}")
    persisted = o3d.io.read_triangle_mesh(str(output_ply))
    persisted_vertices_m = np.asarray(persisted.vertices, dtype=np.float64)
    persisted_faces = np.asarray(persisted.triangles, dtype=np.int64)
    if not np.array_equal(persisted_faces, faces):
        raise RuntimeError("Simulation PLY changed authoritative topology")
    expected_persisted_normals = _vertex_normals(persisted_vertices_m, persisted_faces)
    ply_has_vertex_normals = persisted.has_vertex_normals()
    maximum_ply_normal_error_degrees: float | None = None
    if ply_has_vertex_normals:
        stored_ply_normals = np.asarray(persisted.vertex_normals, dtype=np.float64)
        normal_dot = np.sum(stored_ply_normals * expected_persisted_normals, axis=1)
        maximum_ply_normal_error_degrees = float(
            np.degrees(np.arccos(np.clip(normal_dot, -1.0, 1.0))).max()
        )
    if float(reduction_mm) > 0.0 and not ply_has_vertex_normals:
        raise RuntimeError("The simulated PLY was exported without recomputed vertex normals")
    persisted_displacement_mm = np.linalg.norm(persisted_vertices_m - vertices_m, axis=1) * 1000.0
    persistence_error_mm = (
        np.linalg.norm(
            persisted_vertices_m - simulated_vertices_m,
            axis=1,
        )
        * 1000.0
    )
    if float(reduction_mm) == 0.0 and not np.array_equal(persisted_vertices_m, vertices_m):
        raise RuntimeError("A 0.0 mm simulation must have identical vertex positions")
    simulation_identity = geometry_identity(persisted_vertices_m, persisted_faces)
    geometry_hash_changed = simulation_identity["geometry_id"] != source_identity["geometry_id"]
    affected_vertex_count = int(np.count_nonzero(persisted_displacement_mm > 1e-6))
    vertices_moved_over_point_one_mm = int(np.count_nonzero(persisted_displacement_mm > 0.1))
    maximum_displacement_mm = float(np.max(persisted_displacement_mm))
    median_displacement_mm = _positive_median(persisted_displacement_mm)
    roi_export = _export_roi_ply(
        output_roi_ply,
        persisted_vertices_m,
        persisted_faces,
        roi_vertex_mask,
        persisted_displacement_mm,
    )
    roi_metadata["exported_diagnostic_vertex_count"] = roi_export["vertex_count"]
    roi_metadata["exported_diagnostic_triangle_count"] = roi_export["triangle_count"]
    moved_vertex_export_count = _export_moved_vertices_ply(
        output_moved_vertices_ply,
        persisted_vertices_m,
        persisted_displacement_mm,
    )
    maximum_profile_change_mm = _write_profile_svg(output_profile_svg, profile_diagnostic)
    output_profile_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "units": "millimetres",
                "detected_hump_apex": roi_metadata["detected_hump_apex"],
                "anatomical_displacements": roi_metadata["profile_point_diagnostics"],
                **{
                    name: np.asarray(values).round(6).tolist()
                    for name, values in profile_diagnostic.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    frame = _anatomical_frame(landmarks_mm)
    cross_section_diagnostic = _cross_section_diagnostics(
        vertices_m,
        target_vertices_m,
        persisted_vertices_m,
        frame,
        float(roi_metadata["detected_hump_apex"]["normalized_nasion_to_supratip"]),
        float(roi_metadata["lateral_half_width_mm"]),
        float(roi_metadata["profile_core_half_width_mm"]),
        float(roi_metadata["transverse_bridge_core_half_width_mm"]),
    )
    _write_cross_section_svg(output_cross_sections_svg, cross_section_diagnostic)
    output_cross_sections_json.write_text(
        json.dumps(
            {"schema_version": 1, "units": "millimetres", **cross_section_diagnostic}, indent=2
        ),
        encoding="utf-8",
    )
    acceptance_diagnostic = _acceptance_diagnostics(
        profile_diagnostic,
        cross_section_diagnostic,
        frame,
        applied_reduction_mm,
        roi_metadata["vault_displacement_coverage"],
    )
    source_longitudinal, _, _ = _coordinates(vertices_m * 1000.0, frame)
    maximum_displacement_vertex = int(np.argmax(persisted_displacement_mm))
    maximum_displacement_longitudinal_mm = float(source_longitudinal[maximum_displacement_vertex])
    apex_longitudinal_mm = float(roi_metadata["detected_hump_apex"]["longitudinal_mm_from_nasion"])
    maximum_to_apex_distance_mm = abs(maximum_displacement_longitudinal_mm - apex_longitudinal_mm)
    normalized_vertex_longitudinal = (source_longitudinal - float(frame["roi_start_mm"])) / (
        float(frame["roi_end_mm"]) - float(frame["roi_start_mm"])
    )
    supratip_mask = roi_vertex_mask & (normalized_vertex_longitudinal >= 0.78)
    maximum_supratip_displacement_mm = (
        float(np.max(persisted_displacement_mm[supratip_mask])) if np.any(supratip_mask) else 0.0
    )
    render_paths = _write_diagnostic_renders(
        output_dir,
        label,
        vertices_m,
        persisted_vertices_m,
        faces,
        frame,
        persisted_displacement_mm,
        np.asarray(source.vertex_colors) if source.has_vertex_colors() else None,
    )

    if float(reduction_mm) == 0.0:
        if maximum_displacement_mm != 0.0 or geometry_hash_changed:
            raise RuntimeError("A 0.0 mm simulation changed the persisted geometry")
    else:
        minimum_meaningful_displacement_mm = 0.8 * applied_reduction_mm
        if maximum_displacement_mm < minimum_meaningful_displacement_mm:
            raise RuntimeError(
                "Persisted hump displacement is not meaningfully close to the request: "
                f"{maximum_displacement_mm:.3f} mm for {float(reduction_mm):.3f} mm requested"
            )
        if not geometry_hash_changed or vertices_moved_over_point_one_mm == 0:
            raise RuntimeError("The non-zero simulation did not change persisted mesh geometry")
        if maximum_profile_change_mm < 0.5 * applied_reduction_mm:
            raise RuntimeError(
                "The simulated dorsal profile did not visibly separate from the source profile"
            )
        if maximum_to_apex_distance_mm > 0.12 * float(frame["roi_end_mm"]):
            raise RuntimeError("Maximum displacement is not centered on the detected hump apex")
        if maximum_supratip_displacement_mm > 0.5 * maximum_displacement_mm:
            raise RuntimeError(
                "The supratip is receiving excessive displacement relative to the hump"
            )
        if not acceptance_diagnostic["passed"]:
            raise RuntimeError(
                "The simulated geometry did not pass the profile and frontal-vault acceptance "
                f"checks: {acceptance_diagnostic}"
            )

    short_geometry_id = simulation_identity["geometry_id"].split(":", 1)[-1][:12]
    viewer_glb = output_dir / f"reduction_{label}mm_{short_geometry_id}.glb"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "operation": "dorsal_hump_reduction",
        "solver_id": DORSAL_VAULT_SOLVER_ID,
        "solver_module_sha256": _sha256_file(Path(__file__)),
        "status": "visual_simulation_only",
        "disclaimer": (
            "This is an aesthetic visualization, not a prediction of surgical outcome or a "
            "clinically validated treatment plan."
        ),
        "source_geometry_id": geometry["geometry_id"],
        "simulation_geometry_id": simulation_identity["geometry_id"],
        "exact_final_glb_path": str(viewer_glb.resolve()),
        "source_geometry_unchanged": True,
        "requested_reduction_mm": float(reduction_mm),
        "applied_profile_reduction_mm": round(applied_reduction_mm, 6),
        "maximum_actual_vertex_displacement_mm": round(maximum_displacement_mm, 6),
        "affected_vertex_count": affected_vertex_count,
        "total_vertex_count": len(vertices_m),
        "affected_nasal_roi": roi_metadata,
        "deformation": {
            "direction": (
                "coupled posterior-and-medial deformation of one connected dorsal vault"
            ),
            "solve": "single vector-valued constrained biharmonic surface solve",
            "center_strip_algorithm": False,
            "full_vault_constraint_surface": True,
            "normal_based_displacement": False,
            "left_right_symmetrization": False,
            "topology_changed": False,
            "maximum_computed_displacement_before_persistence_mm": round(
                float(np.max(displacement_mm)), 6
            ),
        },
        "diagnostics": {
            "requested_reduction_mm": float(reduction_mm),
            "applied_profile_reduction_mm": round(applied_reduction_mm, 6),
            "mesh_position_units": "metres",
            "landmark_and_displacement_units": "millimetres",
            "millimetres_to_metres_scale": 0.001,
            "roi_vertex_count": int(roi_metadata["candidate_vertex_count"]),
            "maximum_displacement_mm": round(maximum_displacement_mm, 6),
            "median_displacement_mm": round(median_displacement_mm, 6),
            "median_displacement_scope": "vertices moved more than 0.000001 mm",
            "vertices_moved_over_0_1_mm": vertices_moved_over_point_one_mm,
            "exported_moved_vertex_count": moved_vertex_export_count,
            "pointwise_envelope_clipping": roi_metadata["pointwise_envelope_clipping"],
            "deformation_solver": roi_metadata["deformation_solver"],
            "profile_point_displacements": roi_metadata["profile_point_diagnostics"],
            "transverse_cross_sections": cross_section_diagnostic,
            "acceptance": acceptance_diagnostic,
            "ply_normals": {
                "present": ply_has_vertex_normals,
                "recomputed_from_simulated_geometry": bool(float(reduction_mm) > 0.0),
                "maximum_error_degrees": (
                    round(maximum_ply_normal_error_degrees, 6)
                    if maximum_ply_normal_error_degrees is not None
                    else None
                ),
            },
            "source_mesh_hash": source_identity["geometry_id"],
            "output_mesh_hash": simulation_identity["geometry_id"],
            "output_geometry_hash_differs_from_source": geometry_hash_changed,
            "maximum_in_memory_displacement_mm": round(float(np.max(displacement_mm)), 6),
            "maximum_ply_error_from_memory_mm": round(float(np.max(persistence_error_mm)), 9),
            "maximum_profile_change_mm": round(maximum_profile_change_mm, 6),
            "detected_hump_apex": roi_metadata["detected_hump_apex"],
            "maximum_displacement_longitudinal_mm_from_nasion": round(
                maximum_displacement_longitudinal_mm, 6
            ),
            "maximum_displacement_distance_from_apex_mm": round(maximum_to_apex_distance_mm, 6),
            "maximum_supratip_displacement_mm": round(maximum_supratip_displacement_mm, 6),
        },
        "output_paths": {
            "ply": output_ply.name,
            "glb": output_glb.name,
            "viewer_glb": viewer_glb.name,
            "exact_final_glb": str(viewer_glb.resolve()),
            "affected_roi_ply": output_roi_ply.name,
            "moved_vertices_ply": (
                output_moved_vertices_ply.name if moved_vertex_export_count else None
            ),
            "profile_comparison_svg": output_profile_svg.name,
            "profile_curve_json": output_profile_json.name,
            "cross_sections_svg": output_cross_sections_svg.name,
            "cross_sections_json": output_cross_sections_json.name,
            **{name: path.name for name, path in render_paths.items()},
            "manifest": output_manifest.name,
        },
        "output_directory_at_generation": str(output_dir.resolve()),
    }
    vertex_colors = np.asarray(source.vertex_colors) if source.has_vertex_colors() else None
    glb_diagnostics = _export_simulation_glb(
        source_glb_path,
        output_glb,
        vertices_m,
        persisted_vertices_m,
        faces,
        vertex_colors,
        manifest,
        frame,
        float(roi_metadata["transverse_bridge_core_half_width_mm"]),
    )
    if float(reduction_mm) > 0.0 and not glb_diagnostics["geometry_differs_from_source"]:
        raise RuntimeError("The exported GLB contains the original rather than simulated geometry")
    if float(reduction_mm) > 0.0:
        minimum_transverse_change_mm = 0.04 * applied_reduction_mm
        if (
            glb_diagnostics["maximum_transverse_displacement_from_source_mm"]
            < minimum_transverse_change_mm
            or glb_diagnostics["vertices_with_transverse_change_over_0_1_mm"] == 0
        ):
            raise RuntimeError(
                "The exported GLB does not contain a meaningful transverse bridge change"
            )
    manifest["diagnostics"]["glb_export"] = glb_diagnostics
    shutil.copy2(output_glb, viewer_glb)
    source_hashes_after = {str(path.resolve()): _sha256_file(path) for path in source_paths}
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("A source mesh, geometry manifest, landmark file, or visual GLB changed")
    manifest["source_file_sha256"] = source_hashes_after
    manifest["output_file_sha256"] = {
        "ply": _sha256_file(output_ply),
        "glb": _sha256_file(output_glb),
        "viewer_glb": _sha256_file(viewer_glb),
        "affected_roi_ply": _sha256_file(output_roi_ply),
        "profile_comparison_svg": _sha256_file(output_profile_svg),
        "profile_curve_json": _sha256_file(output_profile_json),
        "cross_sections_svg": _sha256_file(output_cross_sections_svg),
        "cross_sections_json": _sha256_file(output_cross_sections_json),
        **{name: _sha256_file(path) for name, path in render_paths.items()},
    }
    if moved_vertex_export_count:
        manifest["output_file_sha256"]["moved_vertices_ply"] = _sha256_file(
            output_moved_vertices_ply
        )
    output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    get_logger().info(
        "Dorsal hump simulation | requested %.1f mm | actual %.3f mm | %d vertices | %s",
        float(reduction_mm),
        manifest["maximum_actual_vertex_displacement_mm"],
        manifest["affected_vertex_count"],
        output_dir,
    )
    get_logger().info(
        "Dorsal diagnostics | ROI %d vertices | median %.3f mm | >0.1 mm %d | hashes differ %s",
        manifest["diagnostics"]["roi_vertex_count"],
        manifest["diagnostics"]["median_displacement_mm"],
        manifest["diagnostics"]["vertices_moved_over_0_1_mm"],
        manifest["diagnostics"]["output_geometry_hash_differs_from_source"],
    )
    apex = manifest["diagnostics"]["detected_hump_apex"]
    get_logger().info(
        "Dorsal hump apex | %.3f mm from nasion | normalized %.3f | convexity %.3f mm",
        apex["longitudinal_mm_from_nasion"],
        apex["normalized_nasion_to_supratip"],
        apex["outward_convexity_mm"],
    )
    get_logger().info(
        "Transverse GLB verification | max %.3f mm | >0.1 mm %d vertices | final %s",
        manifest["diagnostics"]["glb_export"]["maximum_transverse_displacement_from_source_mm"],
        manifest["diagnostics"]["glb_export"]["vertices_with_transverse_change_over_0_1_mm"],
        manifest["exact_final_glb_path"],
    )
    return manifest
