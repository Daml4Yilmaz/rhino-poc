"""Controlled A/B/C/D diagnostics for fused-point-cloud Poisson reconstruction.

Nothing in this module promotes a diagnostic mesh to the authoritative geometry.
The experiment reads the permanent COLMAP stereo-fusion artifact and writes only
to a caller-selected diagnostic directory.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

VARIANT_DEFINITIONS = {
    "A": {
        "label": "original_xyz_colmap_normals",
        "xyz_source": "original COLMAP stereo-fusion point cloud",
        "normal_source": "original COLMAP stereo-fusion normals",
    },
    "B": {
        "label": "original_xyz_recomputed_normals",
        "xyz_source": "original COLMAP stereo-fusion point cloud",
        "normal_source": "explicitly recomputed and oriented normals",
    },
    "C": {
        "label": "filtered_xyz_colmap_normals",
        "xyz_source": "conservatively filtered COLMAP stereo-fusion points",
        "normal_source": "original COLMAP stereo-fusion normals retained by index",
    },
    "D": {
        "label": "filtered_xyz_recomputed_normals",
        "xyz_source": "conservatively filtered COLMAP stereo-fusion points",
        "normal_source": "explicitly recomputed and oriented normals",
    },
}


@dataclass(frozen=True)
class PoissonDiagnosticConfig:
    """Explicit parameters for a reproducible diagnostic run."""

    poisson_depths: tuple[int, ...] = (9,)
    production_poisson_depth: int = 9
    poisson_trim_percent: float = 4.0
    normal_radius_mm: float = 2.5
    normal_max_nn: int = 64
    normal_fast_computation: bool = False
    orientation_neighbors: int = 30
    outlier_filter_neighbors: int = 30
    outlier_filter_std_ratio: float = 3.0
    maximum_removed_percent: float = 5.0
    normal_sample_count: int = 50_000
    normal_neighbor_count: int = 12
    roi_normal_sample_count: int = 10_000
    thickness_sample_count: int = 2_500
    thickness_radius_mm: float = 2.0
    thickness_max_nn: int = 64
    thickness_min_neighbors: int = 12
    random_seed: int = 20260815

    def validate(self) -> None:
        depths = tuple(dict.fromkeys(self.poisson_depths))
        if not depths or any(depth not in (7, 8, 9, 10, 11) for depth in depths):
            raise ValueError("Diagnostic Poisson depths must be selected from 7 through 11")
        if self.production_poisson_depth not in depths:
            raise ValueError("poisson_depths must include production_poisson_depth")
        if not 0.0 <= self.poisson_trim_percent < 100.0:
            raise ValueError("poisson_trim_percent must be in [0, 100)")
        if self.normal_radius_mm <= 0 or self.normal_max_nn < 3:
            raise ValueError("Normal neighborhood settings must be positive")
        if self.orientation_neighbors < 3:
            raise ValueError("orientation_neighbors must be at least 3")
        if self.outlier_filter_neighbors < 3 or self.outlier_filter_std_ratio <= 0:
            raise ValueError("Outlier-filter settings must be positive")
        if not 0.0 <= self.maximum_removed_percent < 100.0:
            raise ValueError("maximum_removed_percent must be in [0, 100)")
        if self.normal_neighbor_count < 1 or self.normal_sample_count < 1:
            raise ValueError("Normal diagnostic sampling settings must be positive")
        if self.thickness_radius_mm <= 0 or self.thickness_max_nn < 3:
            raise ValueError("Thickness neighborhood settings must be positive")
        if not 3 <= self.thickness_min_neighbors <= self.thickness_max_nn:
            raise ValueError("Invalid minimum neighbor count for thickness estimation")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_positions(points: np.ndarray) -> str:
    normalized = np.ascontiguousarray(points, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(normalized.shape).encode("ascii"))
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def _percentiles(values: np.ndarray, percentiles: tuple[int, ...]) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {f"p{value}": None for value in percentiles}
    return {f"p{value}": round(float(np.percentile(finite, value)), 6) for value in percentiles}


def _artifact_record(path: Path, root: Path, *, role: str) -> dict[str, Any]:
    try:
        relative_path = str(path.relative_to(root))
    except ValueError:
        relative_path = None
    return {
        "role": role,
        "path": relative_path,
        "absolute_path": str(path.resolve()),
        "file_size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _write_report(report: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _normal_statistics(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    scale_mm_per_unit: float,
    sample_count: int,
    neighbor_count: int,
    random_seed: int,
    sample_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure signed and sign-invariant local normal agreement."""
    lengths = np.linalg.norm(normals, axis=1)
    valid = (
        np.all(np.isfinite(points), axis=1)
        & np.all(np.isfinite(normals), axis=1)
        & (lengths > 1e-12)
    )
    candidates = valid if sample_mask is None else valid & np.asarray(sample_mask, dtype=bool)
    candidate_indices = np.flatnonzero(candidates)
    if not len(candidate_indices):
        return {"status": "unavailable", "reason": "No finite ROI points with non-zero normals"}

    valid_indices = np.flatnonzero(valid)
    compact_lookup = np.full(len(points), -1, dtype=np.int64)
    compact_lookup[valid_indices] = np.arange(len(valid_indices), dtype=np.int64)
    compact_candidates = compact_lookup[candidate_indices]
    rng = np.random.default_rng(random_seed)
    selected = rng.choice(
        compact_candidates,
        size=min(sample_count, len(compact_candidates)),
        replace=False,
    )
    valid_points = points[valid]
    unit_normals = normals[valid] / lengths[valid, None]
    query_k = min(neighbor_count + 1, len(valid_points))
    if query_k < 2:
        return {"status": "unavailable", "reason": "Fewer than two valid points"}
    distances, neighbor_indices = cKDTree(valid_points).query(
        valid_points[selected], k=query_k, workers=-1
    )
    distances = np.atleast_2d(distances)[:, 1:]
    neighbor_indices = np.atleast_2d(neighbor_indices)[:, 1:]
    dots = np.einsum("ij,ikj->ik", unit_normals[selected], unit_normals[neighbor_indices])
    dots = np.clip(dots, -1.0, 1.0)
    oriented_angles = np.degrees(np.arccos(dots))
    unoriented_angles = np.degrees(np.arccos(np.abs(dots)))
    return {
        "status": "available",
        "candidate_point_count": len(candidate_indices),
        "sample_count": len(selected),
        "neighbors_per_sample": int(query_k - 1),
        "random_seed": int(random_seed),
        "neighbor_distance_mm": _percentiles(distances * scale_mm_per_unit, (50, 90, 95, 99)),
        "oriented_neighbor_angle_degrees": _percentiles(oriented_angles, (50, 90, 95, 99)),
        "unoriented_plane_angle_degrees": _percentiles(unoriented_angles, (50, 90, 95, 99)),
        "fraction_dot_lt_0": round(float(np.mean(dots < 0.0)), 8),
        "fraction_dot_lt_minus_0_5": round(float(np.mean(dots < -0.5)), 8),
        "fraction_dot_lt_minus_0_9": round(float(np.mean(dots < -0.9)), 8),
        "normal_length": _percentiles(lengths[valid], (1, 50, 99)),
    }


def _sphere_mask(points_mm: np.ndarray, center_mm: np.ndarray, radius_mm: float) -> np.ndarray:
    return np.linalg.norm(points_mm - center_mm, axis=1) <= radius_mm


def _segment_mask(
    points_mm: np.ndarray,
    first_mm: np.ndarray,
    second_mm: np.ndarray,
    radius_mm: float,
    parameter_range: tuple[float, float],
) -> np.ndarray:
    axis = second_mm - first_mm
    squared_length = float(np.dot(axis, axis))
    if squared_length <= 1e-12:
        return np.zeros(len(points_mm), dtype=bool)
    parameters = ((points_mm - first_mm) @ axis) / squared_length
    closest = first_mm + np.clip(parameters, 0.0, 1.0)[:, None] * axis
    distances = np.linalg.norm(points_mm - closest, axis=1)
    return (
        (parameters >= parameter_range[0])
        & (parameters <= parameter_range[1])
        & (distances <= radius_mm)
    )


def _diagnostic_rois(
    points: np.ndarray,
    landmarks_path: Path | None,
    scale_mm_per_unit: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Create explicitly documented, non-clinical landmark-relative ROIs."""
    if landmarks_path is None or not landmarks_path.is_file():
        return {}, {
            "status": "unavailable",
            "reason": "landmarks.json was not supplied; no anatomical ROIs were invented",
        }
    document = json.loads(landmarks_path.read_text(encoding="utf-8"))
    raw_landmarks = document.get("landmarks", {})
    required = {
        "glabella",
        "nasion",
        "pronasale",
        "subnasale",
        "labiale_superius",
        "left_alare",
        "right_alare",
        "left_endocanthion",
        "right_endocanthion",
    }
    missing = sorted(required - raw_landmarks.keys())
    if missing:
        return {}, {
            "status": "unavailable",
            "reason": f"Required diagnostic landmarks are missing: {missing}",
        }
    landmarks = {name: np.asarray(value, dtype=np.float64) for name, value in raw_landmarks.items()}
    points_mm = points * scale_mm_per_unit
    glabella = landmarks["glabella"]
    subnasale = landmarks["subnasale"]
    superior = glabella - subnasale
    superior /= np.linalg.norm(superior)
    eye_midpoint = (landmarks["left_endocanthion"] + landmarks["right_endocanthion"]) / 2.0

    left_anchor = (landmarks["left_endocanthion"] + landmarks["left_alare"]) / 2.0
    right_anchor = (landmarks["right_endocanthion"] + landmarks["right_alare"]) / 2.0
    left_lateral = left_anchor - eye_midpoint
    right_lateral = right_anchor - eye_midpoint
    left_lateral /= np.linalg.norm(left_lateral)
    right_lateral /= np.linalg.norm(right_lateral)

    centers = {
        "central_forehead": glabella + superior * 14.0,
        "left_cheek": left_anchor + left_lateral * 12.0,
        "right_cheek": right_anchor + right_lateral * 12.0,
        "nasal_tip": landmarks["pronasale"],
    }
    masks = {
        "central_forehead": _sphere_mask(points_mm, centers["central_forehead"], 10.0),
        "left_cheek": _sphere_mask(points_mm, centers["left_cheek"], 10.0),
        "right_cheek": _sphere_mask(points_mm, centers["right_cheek"], 10.0),
        "nasal_tip": _sphere_mask(points_mm, centers["nasal_tip"], 6.0),
        "nasal_dorsum": _segment_mask(
            points_mm,
            landmarks["nasion"],
            landmarks["pronasale"],
            radius_mm=4.0,
            parameter_range=(0.15, 0.75),
        ),
    }
    definitions = {
        "status": "available",
        "purpose": "reconstruction diagnostics only; not clinically validated segmentation",
        "coordinate_units": "millimetres",
        "regions": {
            "central_forehead": {
                "shape": "sphere",
                "radius_mm": 10.0,
                "center_definition": "glabella shifted 14 mm along glabella-minus-subnasale",
            },
            "left_cheek": {
                "shape": "sphere",
                "radius_mm": 10.0,
                "center_definition": "left eye/alar midpoint shifted 12 mm away from midline",
            },
            "right_cheek": {
                "shape": "sphere",
                "radius_mm": 10.0,
                "center_definition": "right eye/alar midpoint shifted 12 mm away from midline",
            },
            "nasal_dorsum": {
                "shape": "segment tube",
                "radius_mm": 4.0,
                "segment_fraction": [0.15, 0.75],
                "segment": "nasion to pronasale",
            },
            "nasal_tip": {
                "shape": "sphere",
                "radius_mm": 6.0,
                "center_definition": "pronasale",
            },
            "chin": {
                "status": "unavailable",
                "reason": "The current landmark schema has no menton or pogonion; a chin ROI was not invented",
            },
        },
    }
    return masks, definitions


def _surface_thickness_statistics(
    points: np.ndarray,
    roi_mask: np.ndarray,
    *,
    scale_mm_per_unit: float,
    radius_mm: float,
    max_nn: int,
    min_neighbors: int,
    sample_count: int,
    random_seed: int,
) -> dict[str, Any]:
    """Estimate local plane residuals without moving or smoothing any point."""
    finite = np.all(np.isfinite(points), axis=1)
    valid_indices = np.flatnonzero(finite)
    candidates = np.flatnonzero(finite & np.asarray(roi_mask, dtype=bool))
    if not len(candidates):
        return {"status": "unavailable", "reason": "ROI contains no finite points"}
    compact_lookup = np.full(len(points), -1, dtype=np.int64)
    compact_lookup[valid_indices] = np.arange(len(valid_indices), dtype=np.int64)
    compact_candidates = compact_lookup[candidates]
    rng = np.random.default_rng(random_seed)
    selected = rng.choice(
        compact_candidates,
        size=min(sample_count, len(compact_candidates)),
        replace=False,
    )
    valid_points = points[finite]
    query_k = min(max_nn, len(valid_points))
    distances, neighbor_indices = cKDTree(valid_points).query(
        valid_points[selected], k=query_k, workers=-1
    )
    distances = np.atleast_2d(distances)
    neighbor_indices = np.atleast_2d(neighbor_indices)
    radius_units = radius_mm / scale_mm_per_unit
    signed_residuals: list[np.ndarray] = []
    accepted_neighborhoods = 0
    for row in range(len(selected)):
        local_indices = neighbor_indices[row][distances[row] <= radius_units]
        if len(local_indices) < min_neighbors:
            continue
        local = valid_points[local_indices]
        centered = local - np.mean(local, axis=0)
        covariance = centered.T @ centered / max(len(centered) - 1, 1)
        _, eigenvectors = np.linalg.eigh(covariance)
        signed_residuals.append(centered @ eigenvectors[:, 0] * scale_mm_per_unit)
        accepted_neighborhoods += 1
    if not signed_residuals:
        return {
            "status": "unavailable",
            "reason": "No sampled point had the configured minimum local support",
            "candidate_point_count": len(candidates),
        }
    signed = np.concatenate(signed_residuals)
    absolute = np.abs(signed)
    median_signed = float(np.median(signed))
    return {
        "status": "available",
        "candidate_point_count": len(candidates),
        "sampled_center_count": len(selected),
        "accepted_neighborhood_count": int(accepted_neighborhoods),
        "aggregate_residual_count": len(signed),
        "random_seed": int(random_seed),
        "method": "PCA tangent plane fitted independently in each local neighborhood",
        "radius_mm": float(radius_mm),
        "max_nn": int(max_nn),
        "minimum_neighbors": int(min_neighbors),
        "median_absolute_local_plane_residual_mm": round(float(np.median(absolute)), 6),
        "signed_residual_mad_mm": round(float(np.median(np.abs(signed - median_signed))), 6),
        "absolute_local_plane_residual_mm": _percentiles(absolute, (50, 90, 95, 99)),
        "curvature_caution": (
            "Residuals include real curvature within the configured radius; compare regions and "
            "variants rather than treating this as a direct anatomical error bound"
        ),
    }


def _recompute_and_orient_normals(
    point_cloud,
    *,
    reference_normals: np.ndarray,
    scale_mm_per_unit: float,
    config: PoissonDiagnosticConfig,
) -> tuple[Any, dict[str, Any]]:
    import open3d as o3d

    result = o3d.geometry.PointCloud(point_cloud)
    original_positions_hash = _sha256_positions(np.asarray(result.points))
    radius_units = config.normal_radius_mm / scale_mm_per_unit
    result.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_units,
            max_nn=config.normal_max_nn,
        ),
        fast_normal_computation=config.normal_fast_computation,
    )
    result.orient_normals_consistent_tangent_plane(config.orientation_neighbors)
    normals = np.asarray(result.normals)
    lengths = np.linalg.norm(normals, axis=1)
    valid = (
        np.all(np.isfinite(normals), axis=1)
        & np.all(np.isfinite(reference_normals), axis=1)
        & (lengths > 1e-12)
        & (np.linalg.norm(reference_normals, axis=1) > 1e-12)
    )
    reference_unit = (
        reference_normals[valid] / np.linalg.norm(reference_normals[valid], axis=1)[:, None]
    )
    recomputed_unit = normals[valid] / lengths[valid, None]
    median_reference_dot = float(np.median(np.einsum("ij,ij->i", reference_unit, recomputed_unit)))
    global_sign_flipped = median_reference_dot < 0.0
    if global_sign_flipped:
        normals *= -1.0
        result.normals = o3d.utility.Vector3dVector(normals)
    if _sha256_positions(np.asarray(result.points)) != original_positions_hash:
        raise RuntimeError("Normal recomputation unexpectedly changed point positions")
    settings = {
        "estimator": "Open3D PointCloud.estimate_normals",
        "search": "KDTreeSearchParamHybrid",
        "radius_mm": float(config.normal_radius_mm),
        "radius_in_reconstruction_units": float(radius_units),
        "max_nn": int(config.normal_max_nn),
        "fast_normal_computation": bool(config.normal_fast_computation),
        "orientation_method": (
            "orient_normals_consistent_tangent_plane followed by one global sign alignment "
            "to the median COLMAP-normal direction"
        ),
        "orientation_neighbors": int(config.orientation_neighbors),
        "camera_facing_orientation_used": False,
        "camera_facing_reason": (
            "The standalone fused-cloud experiment does not assume camera poses are reliable; "
            "tangent-plane consistency isolates local orientation"
        ),
        "median_dot_to_reference_before_global_sign_alignment": round(median_reference_dot, 8),
        "global_sign_flipped": bool(global_sign_flipped),
    }
    return result, settings


def _component_summary(mesh) -> dict[str, Any]:
    triangle_count = len(mesh.triangles)
    if not triangle_count:
        return {
            "vertex_count": len(mesh.vertices),
            "triangle_count": 0,
            "triangle_connected_component_count": 0,
            "largest_component_ratio": 0.0,
        }
    _, counts, _ = mesh.cluster_connected_triangles()
    counts_array = np.asarray(counts, dtype=np.int64)
    return {
        "vertex_count": len(mesh.vertices),
        "triangle_count": triangle_count,
        "triangle_connected_component_count": len(counts_array),
        "largest_component_ratio": round(float(counts_array.max() / triangle_count), 8),
    }


def _degenerate_triangle_count(vertices: np.ndarray, triangles: np.ndarray) -> int:
    if not len(triangles):
        return 0
    repeated = (
        (triangles[:, 0] == triangles[:, 1])
        | (triangles[:, 1] == triangles[:, 2])
        | (triangles[:, 2] == triangles[:, 0])
    )
    coordinates = vertices[triangles]
    double_area = np.linalg.norm(
        np.cross(
            coordinates[:, 1] - coordinates[:, 0],
            coordinates[:, 2] - coordinates[:, 0],
        ),
        axis=1,
    )
    extent = float(np.max(np.ptp(vertices, axis=0))) if len(vertices) else 0.0
    tolerance = max(np.finfo(np.float64).eps * max(extent * extent, 1.0) * 100.0, 1e-30)
    return int(np.count_nonzero(repeated | (double_area <= tolerance)))


def _mesh_metrics(mesh, scale_mm_per_unit: float) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    component = _component_summary(mesh)
    if not len(triangles):
        return {
            **component,
            "degenerate_triangle_count": 0,
            "boundary_edge_count": 0,
            "watertight": False,
            "median_edge_length_mm": None,
            "p95_edge_length_mm": None,
        }
    edges = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], axis=0
    )
    edges.sort(axis=1)
    unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
    edge_lengths = (
        np.linalg.norm(vertices[unique_edges[:, 0]] - vertices[unique_edges[:, 1]], axis=1)
        * scale_mm_per_unit
    )
    boundary_edge_count = int(np.count_nonzero(edge_counts == 1))
    nonmanifold_edge_count = int(np.count_nonzero(edge_counts > 2))
    return {
        **component,
        "degenerate_triangle_count": _degenerate_triangle_count(vertices, triangles),
        "boundary_edge_count": boundary_edge_count,
        "nonmanifold_edge_count": nonmanifold_edge_count,
        "watertight": bool(boundary_edge_count == 0 and nonmanifold_edge_count == 0),
        "median_edge_length_mm": round(float(np.median(edge_lengths)), 6),
        "p95_edge_length_mm": round(float(np.percentile(edge_lengths, 95)), 6),
    }


def _keep_largest_component(mesh):
    import open3d as o3d

    labels, counts, _ = mesh.cluster_connected_triangles()
    counts_array = np.asarray(counts, dtype=np.int64)
    if not len(counts_array):
        return o3d.geometry.TriangleMesh(mesh)
    result = o3d.geometry.TriangleMesh(mesh)
    result.remove_triangles_by_mask(np.asarray(labels) != int(np.argmax(counts_array)))
    result.remove_unreferenced_vertices()
    return result


def _reconstruct_mesh(
    point_cloud,
    *,
    output_path: Path,
    depth: int,
    trim_percent: float,
    scale_mm_per_unit: float,
) -> tuple[dict[str, Any], Any]:
    import open3d as o3d

    started = time.monotonic()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud, depth=depth
    )
    stages = {"immediately_after_poisson": _component_summary(mesh)}
    density_values = np.asarray(densities, dtype=np.float64)
    density_cutoff = float(np.percentile(density_values, trim_percent))
    mesh.remove_vertices_by_mask(density_values < density_cutoff)
    stages["after_density_trimming"] = _component_summary(mesh)
    mesh.remove_degenerate_triangles()
    stages["after_degenerate_cleanup"] = _component_summary(mesh)
    mesh.remove_duplicated_triangles()
    stages["after_duplicate_cleanup"] = _component_summary(mesh)
    mesh.remove_unreferenced_vertices()
    stages["after_unreferenced_vertex_cleanup"] = _component_summary(mesh)
    mesh = _keep_largest_component(mesh)
    stages["after_keep_largest_component"] = _component_summary(mesh)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output_path), mesh, write_ascii=False):
        raise RuntimeError(f"Failed to write diagnostic mesh: {output_path}")
    persisted = o3d.io.read_triangle_mesh(str(output_path))
    stages["persisted_mesh"] = _component_summary(persisted)
    return {
        **_mesh_metrics(persisted, scale_mm_per_unit),
        "poisson_depth": int(depth),
        "poisson_trim_percent": float(trim_percent),
        "density_cutoff": density_cutoff,
        "processing_stage_components": stages,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }, persisted


def _qa_style_component_count(path: Path) -> int:
    import trimesh

    loaded = trimesh.load(path, force="mesh", process=False)
    labels = trimesh.graph.connected_component_labels(
        loaded.face_adjacency, node_count=len(loaded.faces)
    )
    return len(np.bincount(labels)) if len(labels) else 0


def _inspect_existing_mesh(path: Path, scale_mm_per_unit: float) -> dict[str, Any]:
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(path))
    result = _mesh_metrics(mesh, scale_mm_per_unit)
    result["qa_style_trimesh_face_component_count"] = _qa_style_component_count(path)
    result["path"] = str(path.resolve())
    return result


def _write_point_cloud(path: Path, point_cloud) -> None:
    import open3d as o3d

    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), point_cloud, write_ascii=False, compressed=False):
        raise RuntimeError(f"Failed to write diagnostic point cloud: {path}")


def _load_production_settings(case_dir: Path) -> tuple[float, int]:
    scale_path = case_dir / "scale.json"
    if not scale_path.is_file():
        raise FileNotFoundError(f"Metric scale is required for diagnostics: {scale_path}")
    scale = float(json.loads(scale_path.read_text(encoding="utf-8"))["scale_mm_per_unit"])
    manifest_path = case_dir / "case.json"
    production_depth = 9
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        production_depth = int(
            manifest.get("stages", {})
            .get("mvs", {})
            .get("parameters", {})
            .get("poisson_depth", production_depth)
        )
    return scale, production_depth


def run_poisson_diagnostic(
    case_dir: Path,
    output_dir: Path,
    *,
    config: PoissonDiagnosticConfig | None = None,
) -> Path:
    """Run the non-authoritative A/B/C/D experiment and return its report path."""
    import open3d as o3d

    case_dir = Path(case_dir).resolve()
    output_dir = Path(output_dir).resolve()
    fused_path = case_dir / "face_dense_fused.ply"
    if not fused_path.is_file():
        raise FileNotFoundError(f"Permanent stereo-fusion artifact is missing: {fused_path}")
    if output_dir == case_dir or output_dir == fused_path.parent:
        raise ValueError("Use a dedicated diagnostic subdirectory, not the case root")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(
            f"Diagnostic output directory is not empty: {output_dir}. Use a new run ID."
        )

    scale_mm_per_unit, recorded_production_depth = _load_production_settings(case_dir)
    config = config or PoissonDiagnosticConfig(
        poisson_depths=(recorded_production_depth,),
        production_poisson_depth=recorded_production_depth,
    )
    config.validate()
    if config.production_poisson_depth != recorded_production_depth:
        raise ValueError(
            "Configured production_poisson_depth does not match case.json: "
            f"{config.production_poisson_depth} != {recorded_production_depth}"
        )

    report_path = output_dir / "poisson_diagnostic_report.json"
    source_hash_before = _sha256_file(fused_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "controlled_poisson_abcd_diagnostic_v1",
        "status": "running",
        "started_at": _utc_now(),
        "case_identifier": case_dir.name,
        "non_authoritative": True,
        "promotion_policy": (
            "No diagnostic point cloud or mesh is used by measurements, landmarks, geometry IDs, "
            "texture export, or the authoritative reconstruction pipeline"
        ),
        "source": {
            "case_directory": str(case_dir),
            "fused_point_cloud": str(fused_path),
            "fused_sha256_before": source_hash_before,
            "scale_mm_per_unit": scale_mm_per_unit,
            "production_poisson_depth_from_case_manifest": recorded_production_depth,
        },
        "configuration": asdict(config),
        "variant_definitions": VARIANT_DEFINITIONS,
        "artifacts": {"point_clouds": {}, "meshes": {}},
        "normal_statistics": {},
        "roi_definitions": {},
        "roi_normal_statistics": {},
        "roi_surface_thickness": {},
        "mesh_results": {},
        "connected_component_investigation": {},
    }
    _write_report(report, report_path)

    try:
        print(f"Loading permanent fused point cloud: {fused_path}", flush=True)
        original = o3d.io.read_point_cloud(str(fused_path))
        if not len(original.points):
            raise RuntimeError("The fused point cloud is empty or unreadable")
        if not original.has_normals():
            raise RuntimeError(
                "Variant A requires the original COLMAP normals, but the artifact has none"
            )
        points = np.asarray(original.points, dtype=np.float64)
        original_normals = np.asarray(original.normals, dtype=np.float64)
        if not np.all(np.isfinite(points)):
            raise RuntimeError("The fused point cloud contains non-finite XYZ coordinates")
        original_positions_hash = _sha256_positions(points)
        report["source"].update(
            {
                "original_fused_point_count": len(points),
                "original_normals_exist": True,
                "colors_exist": bool(original.has_colors()),
                "positions_sha256_float64": original_positions_hash,
            }
        )

        point_cloud_dir = output_dir / "point_clouds"
        a_path = point_cloud_dir / "A_original_xyz_colmap_normals.ply"
        point_cloud_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fused_path, a_path)
        report["artifacts"]["point_clouds"]["A"] = _artifact_record(
            a_path, output_dir, role="exact byte copy of permanent COLMAP stereo-fusion artifact"
        )

        print("B: recomputing and explicitly orienting normals", flush=True)
        recomputed, normal_settings = _recompute_and_orient_normals(
            original,
            reference_normals=original_normals,
            scale_mm_per_unit=scale_mm_per_unit,
            config=config,
        )
        b_path = point_cloud_dir / "B_original_xyz_recomputed_oriented_normals.ply"
        _write_point_cloud(b_path, recomputed)
        report["artifacts"]["point_clouds"]["B"] = _artifact_record(
            b_path, output_dir, role="original XYZ with recomputed and oriented normals"
        )
        report["normal_estimation_settings"] = normal_settings

        print("C/D: applying conservative statistical outlier filtering", flush=True)
        _, inlier_indices = original.remove_statistical_outlier(
            nb_neighbors=config.outlier_filter_neighbors,
            std_ratio=config.outlier_filter_std_ratio,
        )
        inlier_indices = sorted(int(index) for index in inlier_indices)
        filtered = original.select_by_index(inlier_indices)
        removed_count = len(points) - len(inlier_indices)
        removed_percent = removed_count / len(points) * 100.0
        report["filtering"] = {
            "method": "Open3D statistical outlier removal",
            "position_smoothing_performed": False,
            "input_point_count": len(points),
            "output_point_count": len(inlier_indices),
            "removed_point_count": int(removed_count),
            "removed_percentage": round(float(removed_percent), 6),
            "parameters": {
                "nb_neighbors": int(config.outlier_filter_neighbors),
                "std_ratio": float(config.outlier_filter_std_ratio),
            },
            "maximum_allowed_removed_percentage": float(config.maximum_removed_percent),
        }
        if removed_percent > config.maximum_removed_percent:
            raise RuntimeError(
                f"Conservative filter would remove {removed_percent:.3f}% of points, exceeding "
                f"the explicit {config.maximum_removed_percent:.3f}% safety limit"
            )
        c_path = point_cloud_dir / "C_filtered_xyz_colmap_normals.ply"
        _write_point_cloud(c_path, filtered)
        report["artifacts"]["point_clouds"]["C"] = _artifact_record(
            c_path, output_dir, role="filtered XYZ with indexed original COLMAP normals"
        )

        filtered_reference_normals = np.asarray(filtered.normals, dtype=np.float64)
        filtered_recomputed, filtered_normal_settings = _recompute_and_orient_normals(
            filtered,
            reference_normals=filtered_reference_normals,
            scale_mm_per_unit=scale_mm_per_unit,
            config=config,
        )
        d_path = point_cloud_dir / "D_filtered_xyz_recomputed_oriented_normals.ply"
        _write_point_cloud(d_path, filtered_recomputed)
        report["artifacts"]["point_clouds"]["D"] = _artifact_record(
            d_path, output_dir, role="filtered XYZ with recomputed and oriented normals"
        )
        report["filtered_normal_estimation_settings"] = filtered_normal_settings

        variant_clouds = {
            "A": original,
            "B": recomputed,
            "C": filtered,
            "D": filtered_recomputed,
        }
        report["variant_position_identity"] = {
            key: {
                "positions_sha256_float64": _sha256_positions(np.asarray(cloud.points)),
                "point_count": len(cloud.points),
            }
            for key, cloud in variant_clouds.items()
        }
        if (
            report["variant_position_identity"]["B"]["positions_sha256_float64"]
            != original_positions_hash
        ):
            raise RuntimeError("Variant B does not preserve the exact in-memory original XYZ array")
        if (
            report["variant_position_identity"]["C"]["positions_sha256_float64"]
            != report["variant_position_identity"]["D"]["positions_sha256_float64"]
        ):
            raise RuntimeError("Variants C and D do not share identical filtered XYZ arrays")

        print("Computing global and ROI normal diagnostics", flush=True)
        for index, (key, cloud) in enumerate(variant_clouds.items()):
            report["normal_statistics"][key] = _normal_statistics(
                np.asarray(cloud.points),
                np.asarray(cloud.normals),
                scale_mm_per_unit=scale_mm_per_unit,
                sample_count=config.normal_sample_count,
                neighbor_count=config.normal_neighbor_count,
                random_seed=config.random_seed + index,
            )

        roi_masks, roi_definitions = _diagnostic_rois(
            points, case_dir / "landmarks.json", scale_mm_per_unit
        )
        report["roi_definitions"] = roi_definitions
        for roi_index, (roi_name, roi_mask) in enumerate(roi_masks.items()):
            point_count = int(np.count_nonzero(roi_mask))
            report["roi_normal_statistics"][roi_name] = {"point_count": point_count}
            if point_count < config.normal_neighbor_count + 1:
                report["roi_normal_statistics"][roi_name]["status"] = "insufficient_points"
                report["roi_surface_thickness"][roi_name] = {
                    "status": "unavailable",
                    "reason": "ROI has insufficient points",
                    "point_count": point_count,
                }
                continue
            for variant_index, (key, cloud) in enumerate(variant_clouds.items()):
                cloud_points = np.asarray(cloud.points)
                if key in ("C", "D"):
                    cloud_mask = roi_mask[np.asarray(inlier_indices, dtype=np.int64)]
                else:
                    cloud_mask = roi_mask
                report["roi_normal_statistics"][roi_name][key] = _normal_statistics(
                    cloud_points,
                    np.asarray(cloud.normals),
                    scale_mm_per_unit=scale_mm_per_unit,
                    sample_count=config.roi_normal_sample_count,
                    neighbor_count=config.normal_neighbor_count,
                    random_seed=config.random_seed + roi_index * 10 + variant_index,
                    sample_mask=cloud_mask,
                )
            report["roi_surface_thickness"][roi_name] = _surface_thickness_statistics(
                points,
                roi_mask,
                scale_mm_per_unit=scale_mm_per_unit,
                radius_mm=config.thickness_radius_mm,
                max_nn=config.thickness_max_nn,
                min_neighbors=config.thickness_min_neighbors,
                sample_count=config.thickness_sample_count,
                random_seed=config.random_seed + roi_index,
            )
        _write_report(report, report_path)

        for depth_index, depth in enumerate(config.poisson_depths):
            depth_key = str(depth)
            report["mesh_results"][depth_key] = {}
            report["artifacts"]["meshes"][depth_key] = {}
            phase = (
                "primary production-depth comparison"
                if depth_index == 0 and depth == config.production_poisson_depth
                else "depth sweep"
            )
            print(f"Poisson depth {depth} ({phase}): starting A/B/C/D", flush=True)
            for key, cloud in variant_clouds.items():
                definition = VARIANT_DEFINITIONS[key]
                output_path = (
                    output_dir / "meshes" / f"depth_{depth}" / f"{key}_{definition['label']}.ply"
                )
                print(f"  Variant {key}: {definition['label']}", flush=True)
                metrics, _ = _reconstruct_mesh(
                    cloud,
                    output_path=output_path,
                    depth=depth,
                    trim_percent=config.poisson_trim_percent,
                    scale_mm_per_unit=scale_mm_per_unit,
                )
                metrics.update(
                    {
                        "source_point_cloud_variant": key,
                        "xyz_source": definition["xyz_source"],
                        "normal_source": definition["normal_source"],
                    }
                )
                report["mesh_results"][depth_key][key] = metrics
                report["artifacts"]["meshes"][depth_key][key] = _artifact_record(
                    output_path,
                    output_dir,
                    role=f"non-authoritative Poisson diagnostic variant {key} at depth {depth}",
                )
                _write_report(report, report_path)

        component_investigation = report["connected_component_investigation"]
        raw_mesh_path = case_dir / "face_mesh_raw.ply"
        authoritative_mesh_path = case_dir / "face_geometry.ply"
        if raw_mesh_path.is_file():
            component_investigation["existing_raw_production_mesh"] = _inspect_existing_mesh(
                raw_mesh_path, scale_mm_per_unit
            )
        if authoritative_mesh_path.is_file():
            component_investigation["existing_authoritative_metric_mesh"] = _inspect_existing_mesh(
                authoritative_mesh_path, 1000.0
            )

        anomalies: list[str] = []
        production_results = report["mesh_results"][str(config.production_poisson_depth)]
        for key, metrics in production_results.items():
            stages = metrics["processing_stage_components"]
            if stages["after_keep_largest_component"]["triangle_connected_component_count"] != 1:
                anomalies.append(
                    f"Variant {key}: largest-component cleanup did not produce one component"
                )
            if (
                stages["after_keep_largest_component"]["triangle_connected_component_count"]
                != stages["persisted_mesh"]["triangle_connected_component_count"]
            ):
                anomalies.append(f"Variant {key}: component count changed during PLY persistence")
        for label, metrics in component_investigation.items():
            if not isinstance(metrics, dict):
                continue
            open3d_count = metrics.get("triangle_connected_component_count")
            qa_count = metrics.get("qa_style_trimesh_face_component_count")
            if open3d_count != qa_count:
                anomalies.append(
                    f"{label}: Open3D reports {open3d_count} components while QA-style Trimesh "
                    f"reports {qa_count}"
                )
        component_investigation["anomalies"] = anomalies
        component_investigation["interpretation"] = (
            "The current production order is Poisson, density trimming, degenerate/duplicate "
            "cleanup, unreferenced-vertex cleanup, then largest triangle-connected component. "
            "A persisted count above one after this experiment indicates either a cleanup or "
            "serialization anomaly. A pre-existing production mesh with many components while "
            "the diagnostic baseline ends with one is consistent with a stale or older artifact, "
            "but the report alone does not prove its history."
        )

        source_hash_after = _sha256_file(fused_path)
        report["source"]["fused_sha256_after"] = source_hash_after
        report["source"]["source_artifact_unchanged"] = source_hash_after == source_hash_before
        if source_hash_after != source_hash_before:
            raise RuntimeError(
                "The permanent COLMAP stereo-fusion artifact changed during diagnostics"
            )
        report["interpretation_guide"] = {
            "A_rough_B_better": "Original COLMAP normal field is the leading cause",
            "A_B_rough_C_D_better": "Fused XYZ outliers or local thickness are the leading cause",
            "A_B_C_D_similarly_rough": "Investigate Poisson representation and upstream MVS geometry",
            "A_good_existing_production_rough": (
                "A later production surface-processing or stale-artifact path is degrading geometry"
            ),
        }
        report["status"] = "complete"
        report["completed_at"] = _utc_now()
        _write_report(report, report_path)
        print(f"Diagnostic experiment complete: {report_path}", flush=True)
        return report_path
    except Exception as error:
        report["status"] = "failed"
        report["failed_at"] = _utc_now()
        report["error"] = f"{type(error).__name__}: {error}"
        _write_report(report, report_path)
        raise
