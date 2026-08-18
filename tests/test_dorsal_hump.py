import hashlib
import json
from pathlib import Path

import numpy as np

from poc.logging_utils import configure_logging, get_logger
from poc.pipeline.export import export_glb
from poc.pipeline.geometry import geometry_identity
from poc.simulation.dorsal_hump import (
    compute_dorsal_hump_deformation,
    simulate_dorsal_hump_reduction,
)


def _synthetic_face(
    hump_height_mm: float = 3.2,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    lateral = np.linspace(-16.0, 16.0, 17)
    longitudinal = np.linspace(0.0, 60.0, 61)
    xx, yy = np.meshgrid(lateral, longitudinal)
    baseline = 0.32 * yy
    hump = hump_height_mm * np.exp(-0.5 * ((yy - 22.0) / 6.0) ** 2)
    asymmetry = 0.35 * (xx / 16.0)
    zz = baseline + hump - 0.018 * xx**2 + asymmetry
    vertices_mm = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    faces = []
    columns = len(lateral)
    for row in range(len(longitudinal) - 1):
        for column in range(columns - 1):
            lower_left = row * columns + column
            lower_right = lower_left + 1
            upper_left = lower_left + columns
            upper_right = upper_left + 1
            faces.append([lower_left, lower_right, upper_left])
            faces.append([lower_right, upper_right, upper_left])
    landmarks = {
        "nasion": np.asarray([0.0, 0.0, 0.0]),
        "pronasale": np.asarray([0.0, 50.0, 25.0]),
        "subnasale": np.asarray([0.0, 60.0, 0.0]),
        "left_alare": np.asarray([-18.0, 46.0, 8.0]),
        "right_alare": np.asarray([18.0, 46.0, 8.0]),
    }
    return vertices_mm / 1000.0, np.asarray(faces, dtype=np.int64), landmarks


def _synthetic_broad_hump() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    vertices, _, landmarks = _synthetic_face()
    vertices_mm = vertices * 1000.0
    longitudinal = vertices_mm[:, 1]
    phase = np.clip(longitudinal / 42.0, 0.0, 1.0)
    broad_hump = 3.2 * np.sin(np.pi * phase) ** 2
    broad_hump[longitudinal > 42.0] = 0.0
    vertices_mm[:, 2] = (
        0.32 * longitudinal
        + broad_hump
        - 0.018 * vertices_mm[:, 0] ** 2
        + 0.35 * (vertices_mm[:, 0] / 16.0)
    )
    return vertices_mm / 1000.0, landmarks


def _write_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    mesh.compute_vertex_normals()
    assert o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(tmp_path: Path) -> tuple[Path, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    vertices, faces, landmarks = _synthetic_face()
    case = tmp_path / "case"
    case.mkdir()
    mesh_path = case / "face_geometry.ply"
    geometry_path = case / "geometry.json"
    landmarks_path = case / "landmarks.json"
    glb_path = case / "face_model.glb"
    _write_mesh(mesh_path, vertices, faces)

    import open3d as o3d

    persisted = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(persisted.vertices)
    faces = np.asarray(persisted.triangles)
    identity = geometry_identity(vertices, faces)
    geometry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                **identity,
                "source_scale_mm_per_unit": 1.0,
            }
        ),
        encoding="utf-8",
    )
    landmarks_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "geometry_id": identity["geometry_id"],
                "units": "millimetres",
                "landmarks": {name: value.tolist() for name, value in landmarks.items()},
            }
        ),
        encoding="utf-8",
    )
    export_glb(mesh_path, geometry_path, glb_path)
    return case, vertices, faces, landmarks


def test_zero_reduction_is_exactly_identical() -> None:
    vertices, _, landmarks = _synthetic_face()

    simulated, displacement_mm, _ = compute_dorsal_hump_deformation(vertices, landmarks, 0.0)

    assert np.array_equal(simulated, vertices)
    assert np.array_equal(displacement_mm, np.zeros(len(vertices)))


def test_reduction_flattens_only_dorsal_roi_and_preserves_tip() -> None:
    vertices, _, landmarks = _synthetic_face()

    simulated, displacement_mm, roi = compute_dorsal_hump_deformation(vertices, landmarks, 2.0)
    original_mm = vertices * 1000.0
    simulated_mm = simulated * 1000.0

    assert 1.8 <= float(displacement_mm.max()) <= 2.000001
    assert np.count_nonzero(displacement_mm) > 0
    np.testing.assert_allclose(simulated_mm[:, :2], original_mm[:, :2], atol=1e-12)
    assert np.all(displacement_mm[original_mm[:, 1] >= 42.0] == 0.0)
    assert np.all(displacement_mm[np.abs(original_mm[:, 0]) >= roi["lateral_half_width_mm"]] == 0.0)
    center = np.argmin(np.abs(original_mm[:, 0]) + np.abs(original_mm[:, 1] - 22.0))
    assert simulated_mm[center, 2] < original_mm[center, 2]


def test_broad_hump_is_not_absorbed_into_target_profile() -> None:
    vertices, landmarks = _synthetic_broad_hump()

    _, displacement_mm, roi = compute_dorsal_hump_deformation(vertices, landmarks, 2.0)

    assert 1.9 <= float(displacement_mm.max()) <= 2.000001
    assert roi["profile_model"]["available_hump_height_mm"] >= 2.5
    assert "proximal and distal dorsal anchors" in roi["profile_model"]["target_profile"]


def test_slider_sets_peak_reduction_instead_of_detected_hump_height() -> None:
    vertices, _, landmarks = _synthetic_face(hump_height_mm=0.5)

    _, displacement_mm, roi = compute_dorsal_hump_deformation(vertices, landmarks, 5.0)

    assert roi["profile_model"]["available_hump_height_mm"] < 1.0
    assert roi["profile_model"]["available_hump_height_is_clinical_measurement"] is False
    assert 4.8 <= float(displacement_mm.max()) <= 5.000001
    assert "slider value defines peak reduction" in roi["profile_model"]["requested_peak_policy"]


def test_simulation_outputs_are_separate_and_sources_remain_unchanged(
    tmp_path: Path, capsys
) -> None:
    import open3d as o3d

    configure_logging()
    case, source_vertices, source_faces, _ = _case(tmp_path)
    source_paths = [
        case / "face_geometry.ply",
        case / "geometry.json",
        case / "landmarks.json",
        case / "face_model.glb",
    ]
    hashes_before = {path: _sha256(path) for path in source_paths}
    output = case / "simulations" / "dorsal_hump"

    manifest = simulate_dorsal_hump_reduction(
        case / "face_geometry.ply",
        case / "geometry.json",
        case / "landmarks.json",
        output,
        reduction_mm=2.0,
        source_glb_path=case / "face_model.glb",
    )

    assert {path: _sha256(path) for path in source_paths} == hashes_before
    assert manifest["operation"] == "dorsal_hump_reduction"
    assert manifest["requested_reduction_mm"] == 2.0
    assert manifest["source_geometry_unchanged"] is True
    assert manifest["affected_vertex_count"] > 0
    assert 0.0 < manifest["maximum_actual_vertex_displacement_mm"] <= 2.000001
    assert (output / "reduction_2.0mm.ply").is_file()
    assert (output / "reduction_2.0mm.glb").is_file()
    assert (output / manifest["output_paths"]["viewer_glb"]).is_file()
    assert (output / "reduction_2.0mm_affected_roi.ply").is_file()
    assert (output / "reduction_2.0mm_profile.svg").is_file()
    assert manifest == json.loads((output / "simulation.json").read_text())
    simulated = o3d.io.read_triangle_mesh(str(output / "reduction_2.0mm.ply"))
    np.testing.assert_array_equal(np.asarray(simulated.triangles), source_faces)
    assert not np.array_equal(np.asarray(simulated.vertices), source_vertices)
    captured = capsys.readouterr()
    assert "Logging error" not in captured.err
    assert f"{manifest['affected_vertex_count']} vertices" in captured.err
    assert "Dorsal diagnostics | ROI" in captured.err
    get_logger().handlers.clear()


def test_four_mm_persists_visible_profile_and_exported_geometry_change(tmp_path: Path) -> None:
    import open3d as o3d

    case, source_vertices, _, _ = _case(tmp_path)
    output = case / "simulations" / "dorsal_hump"

    manifest = simulate_dorsal_hump_reduction(
        case / "face_geometry.ply",
        case / "geometry.json",
        case / "landmarks.json",
        output,
        reduction_mm=4.0,
        source_glb_path=case / "face_model.glb",
    )

    diagnostics = manifest["diagnostics"]
    assert diagnostics["requested_reduction_mm"] == 4.0
    assert diagnostics["mesh_position_units"] == "metres"
    assert diagnostics["millimetres_to_metres_scale"] == 0.001
    assert diagnostics["roi_vertex_count"] > 0
    assert 3.8 <= diagnostics["maximum_displacement_mm"] <= 4.000001
    assert diagnostics["median_displacement_mm"] > 0.0
    assert diagnostics["vertices_moved_over_0_1_mm"] > 0
    assert diagnostics["source_mesh_hash"] != diagnostics["output_mesh_hash"]
    assert diagnostics["output_geometry_hash_differs_from_source"] is True
    assert diagnostics["maximum_ply_error_from_memory_mm"] < 1e-6
    assert diagnostics["maximum_profile_change_mm"] >= 3.0
    assert diagnostics["glb_export"]["geometry_differs_from_source"] is True
    assert diagnostics["glb_export"]["maximum_displacement_from_source_mm"] >= 3.8

    simulated = o3d.io.read_triangle_mesh(str(output / manifest["output_paths"]["ply"]))
    simulated_vertices = np.asarray(simulated.vertices)
    assert float(np.max(np.linalg.norm(simulated_vertices - source_vertices, axis=1))) >= 0.0038

    roi_mesh = o3d.io.read_triangle_mesh(str(output / manifest["output_paths"]["affected_roi_ply"]))
    roi_vertices_mm = np.asarray(roi_mesh.vertices) * 1000.0
    assert 0 < len(roi_vertices_mm) < len(source_vertices)
    assert np.max(np.abs(roi_vertices_mm[:, 0])) <= 10.1
    assert np.max(roi_vertices_mm[:, 1]) < 42.0

    profile_svg = (output / manifest["output_paths"]["profile_comparison_svg"]).read_text()
    assert "red: source | blue: simulation" in profile_svg
    viewer_glb = output / manifest["output_paths"]["viewer_glb"]
    assert viewer_glb.is_file()
    assert manifest["output_file_sha256"]["viewer_glb"] == _sha256(viewer_glb)


def test_persisted_zero_reduction_ply_is_byte_identical_to_source(tmp_path: Path) -> None:
    case, _, _, _ = _case(tmp_path)
    source = case / "face_geometry.ply"
    output = case / "simulations" / "dorsal_hump"

    manifest = simulate_dorsal_hump_reduction(
        source,
        case / "geometry.json",
        case / "landmarks.json",
        output,
        reduction_mm=0.0,
        source_glb_path=case / "face_model.glb",
    )

    assert (output / "reduction_0.0mm.ply").read_bytes() == source.read_bytes()
    assert manifest["maximum_actual_vertex_displacement_mm"] == 0.0
    assert manifest["affected_vertex_count"] == 0
    assert manifest["simulation_geometry_id"] == manifest["source_geometry_id"]
    assert manifest["diagnostics"]["maximum_displacement_mm"] == 0.0
    assert manifest["diagnostics"]["vertices_moved_over_0_1_mm"] == 0
    assert manifest["diagnostics"]["output_geometry_hash_differs_from_source"] is False
    assert manifest["diagnostics"]["maximum_profile_change_mm"] == 0.0
    assert (output / manifest["output_paths"]["affected_roi_ply"]).is_file()
