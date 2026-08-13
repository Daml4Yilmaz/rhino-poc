import json

import numpy as np

from poc.pipeline.export import export_glb
from poc.pipeline.geometry import create_authoritative_geometry, geometry_identity
from poc.pipeline.landmarks import _bind_to_surface


def _write_triangle_mesh(path, vertices, triangles) -> None:
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)),
    )
    mesh.compute_vertex_normals()
    assert o3d.io.write_triangle_mesh(str(path), mesh)


def test_authoritative_geometry_scales_without_changing_topology(tmp_path) -> None:
    import open3d as o3d

    raw = tmp_path / "raw.ply"
    authoritative = tmp_path / "face_geometry.ply"
    metadata_path = tmp_path / "geometry.json"
    vertices = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]
    triangles = [[0, 1, 2]]
    _write_triangle_mesh(raw, vertices, triangles)

    metadata = create_authoritative_geometry(raw, 500.0, authoritative, metadata_path)
    persisted = o3d.io.read_triangle_mesh(str(authoritative))
    np.testing.assert_allclose(np.asarray(persisted.vertices), np.asarray(vertices) * 0.5)
    np.testing.assert_array_equal(np.asarray(persisted.triangles), triangles)
    assert metadata["units"] == "metres"
    assert metadata["geometry_id"].startswith("sha256:")
    assert metadata == json.loads(metadata_path.read_text(encoding="utf-8"))


def test_glb_preserves_authoritative_vertices_triangles_and_identity(tmp_path) -> None:
    import open3d as o3d
    import trimesh

    raw = tmp_path / "raw.ply"
    authoritative = tmp_path / "face_geometry.ply"
    metadata_path = tmp_path / "geometry.json"
    glb = tmp_path / "face_model.glb"
    vertices = [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.3, 0.0]]
    triangles = [[0, 1, 2]]
    _write_triangle_mesh(raw, vertices, triangles)
    create_authoritative_geometry(raw, 1000.0, authoritative, metadata_path)

    export_glb(authoritative, metadata_path, glb)

    source = o3d.io.read_triangle_mesh(str(authoritative))
    rendered = trimesh.load_mesh(str(glb), process=False)
    if isinstance(rendered, trimesh.Scene):
        rendered = next(iter(rendered.geometry.values()))
    np.testing.assert_allclose(rendered.vertices, np.asarray(source.vertices), atol=1e-7)
    np.testing.assert_array_equal(rendered.faces, np.asarray(source.triangles))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["render_asset"]["relationship"] == "identical_vertex_and_triangle_topology"
    assert geometry_identity(rendered.vertices, rendered.faces)["geometry_id"] == metadata[
        "geometry_id"
    ]


def test_surface_binding_uses_triangle_and_barycentric_coordinates(tmp_path) -> None:
    mesh_path = tmp_path / "surface.ply"
    vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    _write_triangle_mesh(mesh_path, vertices, [[0, 1, 2]])

    snapped, bindings, distances = _bind_to_surface(
        {"tip": np.asarray([0.25, 0.25, 0.1])}, mesh_path
    )

    np.testing.assert_allclose(snapped["tip"], [0.25, 0.25, 0.0], atol=1e-6)
    assert bindings["tip"]["triangle_index"] == 0
    np.testing.assert_allclose(bindings["tip"]["barycentric"], [0.5, 0.25, 0.25], atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(bindings["tip"]["barycentric"]) @ vertices,
        bindings["tip"]["position_m"],
    )
    np.testing.assert_allclose(distances["tip"], 100.0, atol=1e-4)


def test_textured_glb_keeps_identity_triangle_mapping_across_uv_seams(tmp_path) -> None:
    import trimesh
    from PIL import Image

    raw = tmp_path / "raw.ply"
    authoritative = tmp_path / "face_geometry.ply"
    metadata_path = tmp_path / "geometry.json"
    textured = tmp_path / "mesh.ply"
    glb = tmp_path / "face_model.glb"
    vertices = [
        [0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.0, 0.3, 0.0],
        [0.2, 0.3, 0.0],
    ]
    _write_triangle_mesh(raw, vertices, [[0, 1, 2], [1, 3, 2]])
    create_authoritative_geometry(raw, 1000.0, authoritative, metadata_path)
    Image.new("RGB", (4, 4), (180, 120, 100)).save(tmp_path / "texture.png")
    textured.write_text(
        """ply
format ascii 1.0
comment TextureFile texture.png
element vertex 4
property float x
property float y
property float z
element face 2
property list uchar int vertex_indices
property list uchar float texcoord
end_header
0 0 0
0.2 0 0
0 0.3 0
0.2 0.3 0
3 0 1 2 6 0 0 1 0 0 1
3 1 3 2 6 1 0 1 1 0 1
""",
        encoding="utf-8",
    )

    export_glb(authoritative, metadata_path, glb, textured_mesh_path=textured)

    rendered = trimesh.load_mesh(str(glb), process=False)
    assert rendered.visual.kind == "texture"
    assert rendered.visual.material.baseColorTexture is not None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["render_asset"]["relationship"] == (
        "same_triangle_order_with_uv_seam_vertex_duplication"
    )
    assert metadata["render_asset"]["triangle_mapping"] == (
        "render_face_index_equals_authoritative_triangle_index"
    )
