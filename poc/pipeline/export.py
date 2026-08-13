"""Export a metric, vertex-colored facial mesh as GLB."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from poc.logging_utils import get_logger


def export_glb(mesh_path: Path, scale_mm_per_unit: float, output_glb: Path) -> Path:
    import open3d as o3d
    import trimesh

    source = o3d.io.read_triangle_mesh(str(mesh_path))
    if not len(source.vertices):
        raise RuntimeError(f"Mesh is empty or unreadable: {mesh_path}")

    # glTF uses metres by convention. Measurements are converted to millimetres
    # separately, but the viewer asset should remain interoperable.
    vertices_m = np.asarray(source.vertices, dtype=np.float64) * (scale_mm_per_unit / 1000.0)
    faces = np.asarray(source.triangles, dtype=np.int64)
    vertex_colors = None
    if source.has_vertex_colors():
        rgb = np.clip(np.asarray(source.vertex_colors) * 255.0, 0, 255).astype(np.uint8)
        alpha = np.full((len(rgb), 1), 255, dtype=np.uint8)
        vertex_colors = np.concatenate([rgb, alpha], axis=1)

    mesh = trimesh.Trimesh(
        vertices=vertices_m,
        faces=faces,
        vertex_colors=vertex_colors,
        process=False,
    )
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_glb))
    extents_mm = mesh.bounding_box.extents * 1000.0
    get_logger().info(
        "GLB export complete | %s | bounding box %.1f × %.1f × %.1f mm | %s",
        output_glb,
        extents_mm[0],
        extents_mm[1],
        extents_mm[2],
        "vertex color" if vertex_colors is not None else "geometry only",
    )
    return output_glb
