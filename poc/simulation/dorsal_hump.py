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
    proximal_anchor = (normalized >= 0.06) & (normalized <= 0.18)
    distal_anchor = (normalized >= 0.82) & (normalized <= 0.94)
    proximal_position = float(np.median(centers[proximal_anchor]))
    distal_position = float(np.median(centers[distal_anchor]))
    proximal_height = float(np.median(profile[proximal_anchor]))
    distal_height = float(np.median(profile[distal_anchor]))
    target_slope = (distal_height - proximal_height) / (distal_position - proximal_position)
    target = proximal_height + target_slope * (centers - proximal_position)
    return centers, profile, target


def _smooth_compact_falloff(normalized_distance: np.ndarray) -> np.ndarray:
    inside = np.clip(1.0 - normalized_distance, 0.0, 1.0)
    return inside * inside * (3.0 - 2.0 * inside)


def compute_dorsal_hump_deformation(
    vertices_m: np.ndarray,
    landmarks_mm: dict[str, np.ndarray],
    reduction_mm: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return simulated vertices and displacement magnitudes without mutating input arrays."""
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
    if reduction_mm == 0.0:
        return vertices_m.copy(), np.zeros(len(vertices_m), dtype=np.float64), roi_metadata

    centers, profile, target = _dorsal_profile(longitudinal, lateral, anterior, frame)
    start = float(frame["roi_start_mm"])
    end = float(frame["roi_end_mm"])
    normalized_centers = (centers - start) / (end - start)
    boundary_falloff = np.sin(np.pi * normalized_centers) ** 2
    convex_excess = np.maximum(profile - target, 0.0) * boundary_falloff
    available_hump_mm = float(np.max(convex_excess))
    if available_hump_mm <= 1e-6:
        centerline_reduction = np.zeros_like(convex_excess)
    else:
        centerline_reduction = convex_excess * min(reduction_mm / available_hump_mm, 1.0)

    profile_at_vertex = np.interp(longitudinal, centers, profile)
    reduction_at_vertex = np.interp(
        longitudinal,
        centers,
        centerline_reduction,
        left=0.0,
        right=0.0,
    )
    lateral_distance = np.abs(lateral) / float(frame["half_width_mm"])
    lateral_weight = _smooth_compact_falloff(lateral_distance)
    depth_gap = np.maximum(profile_at_vertex - anterior, 0.0)
    surface_weight = _smooth_compact_falloff(depth_gap / float(frame["surface_depth_mm"]))
    within_longitudinal = (longitudinal > start) & (longitudinal < end)
    displacement_mm = reduction_at_vertex * lateral_weight * surface_weight * within_longitudinal
    displacement_mm[displacement_mm < 1e-8] = 0.0

    simulated = (
        vertices_m - displacement_mm[:, None] * np.asarray(frame["anterior"])[None, :] / 1000.0
    )
    roi_metadata["profile_model"] = {
        "observed_profile": "smoothed 90th-percentile midline anterior envelope",
        "target_profile": "straight profile through robust proximal and distal dorsal anchors",
        "convex_hump_only": True,
        "available_hump_height_mm": round(available_hump_mm, 6),
    }
    return simulated, displacement_mm, roi_metadata


def _export_simulation_glb(
    source_glb_path: Path | None,
    output_glb_path: Path,
    source_vertices_m: np.ndarray,
    simulated_vertices_m: np.ndarray,
    faces: np.ndarray,
    source_vertex_colors: np.ndarray | None,
    metadata: dict[str, Any],
) -> None:
    import trimesh

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
    simulated_vertices_m, displacement_mm, roi_metadata = compute_dorsal_hump_deformation(
        vertices_m,
        landmarks_mm,
        reduction_mm,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    label = _reduction_label(float(reduction_mm))
    output_ply = output_dir / f"reduction_{label}mm.ply"
    output_glb = output_dir / f"reduction_{label}mm.glb"
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
    persisted_displacement_mm = np.linalg.norm(persisted_vertices_m - vertices_m, axis=1) * 1000.0
    if float(reduction_mm) == 0.0 and not np.array_equal(persisted_vertices_m, vertices_m):
        raise RuntimeError("A 0.0 mm simulation must have identical vertex positions")
    simulation_identity = geometry_identity(persisted_vertices_m, persisted_faces)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "operation": "dorsal_hump_reduction",
        "status": "visual_simulation_only",
        "disclaimer": (
            "This is an aesthetic visualization, not a prediction of surgical outcome or a "
            "clinically validated treatment plan."
        ),
        "source_geometry_id": geometry["geometry_id"],
        "simulation_geometry_id": simulation_identity["geometry_id"],
        "source_geometry_unchanged": True,
        "requested_reduction_mm": float(reduction_mm),
        "maximum_actual_vertex_displacement_mm": round(float(np.max(persisted_displacement_mm)), 6),
        "affected_vertex_count": int(np.count_nonzero(persisted_displacement_mm > 1e-6)),
        "total_vertex_count": len(vertices_m),
        "affected_nasal_roi": roi_metadata,
        "deformation": {
            "direction": "posterior along the landmark-derived sagittal anterior axis",
            "normal_based_displacement": False,
            "left_right_symmetrization": False,
            "topology_changed": False,
            "maximum_computed_displacement_before_persistence_mm": round(
                float(np.max(displacement_mm)), 6
            ),
        },
        "output_paths": {
            "ply": output_ply.name,
            "glb": output_glb.name,
            "manifest": output_manifest.name,
        },
        "output_directory_at_generation": str(output_dir.resolve()),
    }
    vertex_colors = np.asarray(source.vertex_colors) if source.has_vertex_colors() else None
    _export_simulation_glb(
        source_glb_path,
        output_glb,
        vertices_m,
        persisted_vertices_m,
        faces,
        vertex_colors,
        manifest,
    )
    source_hashes_after = {str(path.resolve()): _sha256_file(path) for path in source_paths}
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("A source mesh, geometry manifest, landmark file, or visual GLB changed")
    manifest["source_file_sha256"] = source_hashes_after
    output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    get_logger().info(
        "Dorsal hump simulation | requested %.1f mm | actual %.3f mm | %d vertices | %s",
        float(reduction_mm),
        manifest["maximum_actual_vertex_displacement_mm"],
        manifest["affected_vertex_count"],
        output_dir,
    )
    return manifest
