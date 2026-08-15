"""Export a verified rendering of the authoritative facial geometry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from poc.logging_utils import get_logger
from poc.pipeline.geometry import geometry_identity


def export_glb(
    authoritative_mesh_path: Path,
    geometry_metadata_path: Path,
    output_glb: Path,
    *,
    textured_mesh_path: Path | None = None,
) -> Path:
    import open3d as o3d
    import trimesh

    source = o3d.io.read_triangle_mesh(str(authoritative_mesh_path))
    if not len(source.vertices):
        raise RuntimeError(f"Mesh is empty or unreadable: {authoritative_mesh_path}")

    vertices_m = np.asarray(source.vertices, dtype=np.float64)
    faces = np.asarray(source.triangles, dtype=np.int64)
    metadata = json.loads(geometry_metadata_path.read_text(encoding="utf-8"))
    identity = geometry_identity(vertices_m, faces)
    if identity["geometry_id"] != metadata["geometry_id"]:
        raise RuntimeError("Authoritative mesh does not match geometry metadata")
    vertex_colors = None
    if source.has_vertex_colors():
        rgb = np.clip(np.asarray(source.vertex_colors) * 255.0, 0, 255).astype(np.uint8)
        alpha = np.full((len(rgb), 1), 255, dtype=np.uint8)
        vertex_colors = np.concatenate([rgb, alpha], axis=1)

    authoritative = trimesh.Trimesh(
        vertices=vertices_m,
        faces=faces,
        vertex_colors=vertex_colors,
        process=False,
    )
    mesh = authoritative
    relationship = "identical_vertex_and_triangle_topology"
    role = "authoritative_metric_surface"
    raycast_surface = True
    if textured_mesh_path is not None:
        loaded = trimesh.load_mesh(str(textured_mesh_path), process=False)
        if isinstance(loaded, trimesh.Scene):
            if len(loaded.geometry) != 1:
                raise RuntimeError("Textured PLY must contain exactly one surface")
            loaded = next(iter(loaded.geometry.values()))
        loaded.vertices = np.asarray(loaded.vertices) * (
            metadata["source_scale_mm_per_unit"] / 1000.0
        )
        if len(loaded.faces) != len(faces):
            raise RuntimeError("Texture mapping changed the authoritative triangle count")
        corner_deviation = np.linalg.norm(
            np.asarray(loaded.vertices)[np.asarray(loaded.faces)] - vertices_m[faces], axis=2
        )
        if float(corner_deviation.max()) > 1e-6:
            raise RuntimeError("Texture mapping changed triangle order or surface positions")
        if getattr(loaded.visual, "kind", None) != "texture":
            raise RuntimeError("Textured PLY does not contain usable UV coordinates")
        texture_image = getattr(loaded.visual.material, "image", None)
        if texture_image is None:
            raise RuntimeError("Textured PLY does not reference a texture image")
        skin_material = trimesh.visual.material.PBRMaterial(
            name="registered_skin",
            baseColorTexture=texture_image,
            metallicFactor=0.0,
            roughnessFactor=0.72,
            doubleSided=False,
        )
        loaded.visual = trimesh.visual.TextureVisuals(
            uv=np.asarray(loaded.visual.uv, dtype=np.float64),
            material=skin_material,
        )
        mesh = loaded
        relationship = "same_triangle_order_with_uv_seam_vertex_duplication"
        role = "registered_visual_surface"
        raycast_surface = False
    scene = trimesh.Scene(mesh)

    def add_geometry_metadata(tree: dict) -> None:
        extras = {
            "schema_version": 1,
            "role": role,
            "geometry_id": metadata["geometry_id"],
            "topology_sha256": metadata["topology_sha256"],
            "units": "metres",
            "raycast_surface": raycast_surface,
            "authoritative_surface": authoritative_mesh_path.name,
            "triangle_mapping": "render_face_index_equals_authoritative_triangle_index",
            "visual_displacement_affects_measurements": False,
            "material_model": "pbr_metallic_roughness",
            "metallic_factor": 0.0,
            "roughness_factor": 0.72,
        }
        tree.setdefault("asset", {}).setdefault("extras", {})["rhino_poc"] = extras
        for gltf_mesh in tree.get("meshes", []):
            gltf_mesh["extras"] = extras
            for primitive in gltf_mesh.get("primitives", []):
                primitive["extras"] = extras

    output_glb.parent.mkdir(parents=True, exist_ok=True)
    glb = trimesh.exchange.gltf.export_glb(scene, tree_postprocessor=add_geometry_metadata)
    output_glb.write_bytes(glb)

    rendered = trimesh.load_mesh(str(output_glb), process=False)
    if isinstance(rendered, trimesh.Scene):
        if len(rendered.geometry) != 1:
            raise RuntimeError("GLB verification expected exactly one surface")
        rendered = next(iter(rendered.geometry.values()))
    rendered_vertices = np.asarray(rendered.vertices, dtype=np.float64)
    rendered_faces = np.asarray(rendered.faces, dtype=np.int64)
    if len(rendered_faces) != len(faces):
        raise RuntimeError("GLB export changed the authoritative triangle count")
    rendered_corners = rendered_vertices[rendered_faces]
    authoritative_corners = vertices_m[faces]
    maximum_deviation_m = float(
        np.max(np.linalg.norm(rendered_corners - authoritative_corners, axis=2))
    )
    if maximum_deviation_m > 1e-6:
        raise RuntimeError(
            f"GLB export moved the authoritative surface by {maximum_deviation_m:.3g} m"
        )

    metadata["render_asset"] = {
        "path": output_glb.name,
        "relationship": relationship,
        "vertex_count": len(rendered_vertices),
        "triangle_count": len(rendered_faces),
        "maximum_position_deviation_m": maximum_deviation_m,
        "triangle_mapping": "render_face_index_equals_authoritative_triangle_index",
        "raycast_surface": authoritative_mesh_path.name,
        "material": {
            "model": "pbr_metallic_roughness",
            "metallic_factor": 0.0,
            "roughness_factor": 0.72,
            "base_color_source": "registered_original_rgb_views",
        },
    }
    geometry_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    extents_mm = authoritative.bounding_box.extents * 1000.0
    get_logger().info(
        "GLB export complete | %s | bounding box %.1f × %.1f × %.1f mm | %s",
        output_glb,
        extents_mm[0],
        extents_mm[1],
        extents_mm[2],
        "UV texture atlas"
        if textured_mesh_path is not None
        else ("vertex color" if vertex_colors is not None else "geometry only"),
    )
    return output_glb
