"""Create and identify the authoritative metric facial surface."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def geometry_identity(vertices_m: np.ndarray, faces: np.ndarray) -> dict[str, str]:
    """Return stable identifiers at glTF's float32 positional precision."""
    positions = np.asarray(vertices_m, dtype="<f4")
    triangles = np.asarray(faces, dtype="<u4")
    vertex_hash = _sha256(positions)
    topology_hash = _sha256(triangles)
    geometry_hash = _sha256(positions, triangles)
    return {
        "geometry_id": f"sha256:{geometry_hash}",
        "vertex_positions_sha256": vertex_hash,
        "topology_sha256": topology_hash,
    }


def create_authoritative_geometry(
    raw_mesh_path: Path,
    scale_mm_per_unit: float,
    output_mesh_path: Path,
    output_metadata_path: Path,
) -> dict:
    """Scale a reconstruction into metres without changing vertices or topology."""
    import open3d as o3d

    source = o3d.io.read_triangle_mesh(str(raw_mesh_path))
    vertices = np.asarray(source.vertices, dtype=np.float64)
    faces = np.asarray(source.triangles, dtype=np.int64)
    if not len(vertices) or not len(faces):
        raise RuntimeError(f"Mesh is empty or unreadable: {raw_mesh_path}")
    if not np.isfinite(scale_mm_per_unit) or scale_mm_per_unit <= 0:
        raise ValueError("scale_mm_per_unit must be a finite positive number")

    metric = o3d.geometry.TriangleMesh(source)
    metric.vertices = o3d.utility.Vector3dVector(vertices * (scale_mm_per_unit / 1000.0))
    if not metric.has_vertex_normals():
        metric.compute_vertex_normals()

    output_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output_mesh_path), metric, write_ascii=False):
        raise RuntimeError(f"Failed to write authoritative mesh: {output_mesh_path}")

    # Identity is computed from the persisted asset, which is what all downstream
    # stages consume. Float32 normalization matches glTF positional precision.
    persisted = o3d.io.read_triangle_mesh(str(output_mesh_path))
    persisted_vertices = np.asarray(persisted.vertices, dtype=np.float64)
    persisted_faces = np.asarray(persisted.triangles, dtype=np.int64)
    if not np.array_equal(faces, persisted_faces):
        raise RuntimeError("Authoritative PLY export changed triangle order or topology")
    identity = geometry_identity(persisted_vertices, persisted_faces)

    metadata = {
        "schema_version": 1,
        **identity,
        "role": "authoritative_metric_surface",
        "units": "metres",
        "coordinate_system": "COLMAP world coordinates",
        "vertex_count": len(persisted_vertices),
        "triangle_count": len(persisted_faces),
        "authoritative_mesh": output_mesh_path.name,
        "source_mesh": raw_mesh_path.name,
        "source_scale_mm_per_unit": float(scale_mm_per_unit),
        "measurement_policy": {
            "surface": "authoritative_mesh",
            "landmark_binding": "triangle_index_and_barycentric_coordinates",
        },
        "visual_detail_policy": {
            "albedo": "appearance_only",
            "normal_map": "appearance_only",
            "roughness_map": "appearance_only",
            "displacement": "visual_only_unless_validated_and_promoted_to_a_new_geometry_id",
        },
    }
    output_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
