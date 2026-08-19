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
    lateral = np.linspace(-24.0, 24.0, 25)
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


def _synthetic_broad_hump() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    vertices, faces, landmarks = _synthetic_face()
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
    return vertices_mm / 1000.0, faces, landmarks


def _synthetic_upper_hump_with_larger_supratip_bulge() -> tuple[
    np.ndarray, np.ndarray, dict[str, np.ndarray]
]:
    vertices, faces, landmarks = _synthetic_face(hump_height_mm=0.0)
    vertices_mm = vertices * 1000.0
    lateral = vertices_mm[:, 0]
    longitudinal = vertices_mm[:, 1]
    upper_hump = 2.7 * np.exp(-0.5 * ((longitudinal - 16.0) / 4.5) ** 2)
    lower_bulge = 4.0 * np.exp(-0.5 * ((longitudinal - 35.0) / 3.0) ** 2)
    vertices_mm[:, 2] = 0.32 * longitudinal + upper_hump + lower_bulge - 0.018 * lateral**2
    return vertices_mm / 1000.0, faces, landmarks


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
    vertices, faces, landmarks = _synthetic_face()

    simulated, displacement_mm, _ = compute_dorsal_hump_deformation(vertices, landmarks, 0.0, faces)

    assert np.array_equal(simulated, vertices)
    assert np.array_equal(displacement_mm, np.zeros(len(vertices)))


def test_reduction_flattens_only_dorsal_roi_and_preserves_tip() -> None:
    vertices, faces, landmarks = _synthetic_face()

    simulated, displacement_mm, roi = compute_dorsal_hump_deformation(
        vertices, landmarks, 2.0, faces
    )
    original_mm = vertices * 1000.0
    simulated_mm = simulated * 1000.0

    assert 1.8 <= float(displacement_mm.max()) <= np.hypot(2.0, 0.2) + 1e-6
    assert np.count_nonzero(displacement_mm) > 0
    np.testing.assert_allclose(simulated_mm[:, 1], original_mm[:, 1], atol=1e-12)
    lateral_motion = simulated_mm[:, 0] - original_mm[:, 0]
    assert float(np.max(np.abs(lateral_motion))) <= 0.21
    assert np.all(original_mm[:, 0] * lateral_motion <= 1e-10)
    assert np.all(displacement_mm[original_mm[:, 1] >= 42.0] == 0.0)
    assert np.all(displacement_mm[np.abs(original_mm[:, 0]) >= roi["lateral_half_width_mm"]] == 0.0)
    center = np.argmin(np.abs(original_mm[:, 0]) + np.abs(original_mm[:, 1] - 22.0))
    assert simulated_mm[center, 2] < original_mm[center, 2]


def test_broad_hump_is_not_absorbed_into_target_profile() -> None:
    vertices, faces, landmarks = _synthetic_broad_hump()

    _, displacement_mm, roi = compute_dorsal_hump_deformation(vertices, landmarks, 2.0, faces)

    assert 1.9 <= float(displacement_mm.max()) <= np.hypot(2.0, 0.2) + 1e-6
    assert roi["profile_model"]["available_hump_height_mm"] >= 2.5
    assert "nasion and supratip anchors" in roi["profile_model"]["reference_profile"]
    assert "no-scoop" in roi["profile_model"]["target_profile"]


def test_slider_cannot_create_a_new_sagittal_scoop_below_reference() -> None:
    vertices, faces, landmarks = _synthetic_face(hump_height_mm=0.5)

    _, displacement_mm, roi = compute_dorsal_hump_deformation(vertices, landmarks, 5.0, faces)

    assert roi["profile_model"]["available_hump_height_mm"] < 1.0
    assert roi["profile_model"]["available_hump_height_is_clinical_measurement"] is False
    applied = roi["profile_model"]["applied_peak_reduction_mm"]
    assert applied == roi["profile_model"]["available_hump_height_mm"]
    assert roi["profile_model"]["limited_by_detected_convexity"] is True
    assert roi["profile_model"]["maximum_new_below_reference_mm"] == 0.0
    assert 0.3 <= float(displacement_mm.max()) <= np.hypot(applied, 0.1 * applied) + 1e-6
    assert "upper bound" in roi["profile_model"]["requested_peak_policy"]


def test_upper_hump_apex_is_targeted_instead_of_larger_supratip_bulge() -> None:
    vertices, faces, landmarks = _synthetic_upper_hump_with_larger_supratip_bulge()

    _, displacement_mm, roi = compute_dorsal_hump_deformation(vertices, landmarks, 4.0, faces)

    longitudinal_mm = vertices[:, 1] * 1000.0
    apex = roi["detected_hump_apex"]
    maximum_vertex_longitudinal_mm = float(longitudinal_mm[np.argmax(displacement_mm)])
    lower_region = (longitudinal_mm >= 30.0) & (longitudinal_mm <= 39.0)
    assert 12.0 <= apex["longitudinal_mm_from_nasion"] <= 20.0
    assert 12.0 <= maximum_vertex_longitudinal_mm <= 20.0
    applied = roi["profile_model"]["applied_peak_reduction_mm"]
    assert 1.7 <= applied < 4.0
    assert float(displacement_mm.max()) >= 0.95 * applied
    assert float(displacement_mm[lower_region].max()) < 0.5 * float(displacement_mm.max())


def test_upper_and_mid_hump_shoulders_receive_the_apex_correction_fraction() -> None:
    vertices, faces, landmarks = _synthetic_upper_hump_with_larger_supratip_bulge()

    _, _, roi = compute_dorsal_hump_deformation(vertices, landmarks, 1.0, faces)

    applied_fraction = roi["profile_model"]["applied_upper_mid_convexity_fraction"]
    points = roi["profile_point_diagnostics"]
    assert 0.4 < applied_fraction < 0.5
    for name in ("upper_dorsum", "hump_apex"):
        point = points[name]
        assert point["source_convex_excess_mm"] > 0.1
        assert abs(point["requested_convexity_reduction_fraction"] - applied_fraction) < 0.01
        assert point["final_posterior_displacement_mm"] > (
            0.9 * point["requested_posterior_displacement_mm"]
        )
    assert 0.0 < points["mid_dorsum"]["requested_convexity_reduction_fraction"] < applied_fraction
    assert points["lower_dorsum"]["requested_posterior_displacement_mm"] == 0.0
    assert points["supratip"]["requested_posterior_displacement_mm"] == 0.0
    assert points["pronasale"]["final_posterior_displacement_mm"] == 0.0


def test_connected_profile_vertex_is_not_left_fixed_by_pointwise_clipping() -> None:
    vertices, faces, landmarks = _synthetic_face()
    vertices_mm = vertices * 1000.0
    depressed = np.argmin(np.abs(vertices_mm[:, 0]) + np.abs(vertices_mm[:, 1] - 22.0))
    vertices[depressed, 2] -= 0.002

    _, displacement_mm, roi = compute_dorsal_hump_deformation(vertices, landmarks, 4.0, faces)

    assert roi["pointwise_envelope_clipping"] is False
    solver = roi["deformation_solver"]
    assert solver["method"] == "coupled_vector_biharmonic_dorsal_vault"
    assert solver["constraint_vertex_count"] > solver["fixed_boundary_vertex_count"]
    assert roi["transverse_model"]["entire_vault_is_constraint_surface"] is True
    assert displacement_mm[depressed] > 2.8


def test_profile_correction_is_continuous_without_a_scooped_section() -> None:
    vertices, faces, landmarks = _synthetic_face()

    _, displacement_mm, roi = compute_dorsal_hump_deformation(vertices, landmarks, 4.0, faces)

    vertices_mm = vertices * 1000.0
    midline = np.abs(vertices_mm[:, 0]) < 0.1
    order = np.argsort(vertices_mm[midline, 1])
    ridge_displacement = displacement_mm[midline][order]
    active = ridge_displacement > 0.05
    active_values = ridge_displacement[active]
    assert len(active_values) > 20
    assert np.max(np.abs(np.diff(active_values))) < 0.65
    assert np.max(np.abs(np.diff(active_values, n=2))) < 0.25
    solver = roi["deformation_solver"]
    assert solver["constraint_vertex_count"] > 0
    assert solver["fixed_boundary_vertex_count"] > 0
    assert solver["p95_neighbor_displacement_change_mm"] < 1.1
    assert solver["maximum_neighbor_displacement_change_mm"] < 1.8


def test_hump_cross_section_remains_convex_and_blends_into_sidewalls() -> None:
    vertices, faces, landmarks = _synthetic_face()

    simulated, displacement_mm, roi = compute_dorsal_hump_deformation(
        vertices, landmarks, 4.0, faces
    )

    source_mm = vertices * 1000.0
    simulated_mm = simulated * 1000.0
    hump_level = np.abs(source_mm[:, 1] - 22.0) < 0.1
    lateral = simulated_mm[hump_level, 0]
    anterior = simulated_mm[hump_level, 2]
    center = np.abs(lateral) <= 0.1
    central_bridge = np.abs(lateral) <= 6.0
    sidewall = (np.abs(lateral) >= 10.0) & (np.abs(lateral) <= 18.0)
    curvature = float(np.polyfit(lateral[central_bridge], anterior[central_bridge], 2)[0])

    assert curvature < -0.015
    assert float(np.min(anterior[center])) > float(np.max(anterior[sidewall])) + 0.8
    source_sidewall = (
        hump_level & (np.abs(source_mm[:, 0]) >= 6.0) & (np.abs(source_mm[:, 0]) <= 10.0)
    )
    assert float(np.min(displacement_mm[source_sidewall])) > 2.5
    outer_sidewall = (
        hump_level & (np.abs(source_mm[:, 0]) >= 10.0) & (np.abs(source_mm[:, 0]) <= 14.0)
    )
    center_displacement = float(
        np.median(displacement_mm[hump_level & (np.abs(source_mm[:, 0]) <= 0.1)])
    )
    assert float(np.median(displacement_mm[outer_sidewall])) > 0.75 * center_displacement
    assert roi["transverse_model"]["coupled_single_solve"] is True
    assert roi["transverse_model"]["slope_target_weight_at_normalized_radius_0_5"] > 0.9
    assert roi["transverse_model"]["sidewall_target_weight_at_normalized_radius_0_8"] > 0.6
    assert roi["transverse_model"]["actual_maximum_medial_adjustment_mm"] > 0.25
    assert roi["transverse_model"]["actual_maximum_medial_adjustment_mm"] <= 0.6
    coverage = roi["vault_displacement_coverage"]
    assert coverage["center_strip_pattern_detected"] is False
    assert coverage["slope_to_ridge_median_ratio"] > 0.9
    assert coverage["sidewall_to_ridge_median_ratio"] > 0.7
    assert coverage["sidewalls"]["fraction_moved_over_0_1_mm"] > 0.9


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
    assert manifest["solver_id"] == "coupled-vector-biharmonic-full-vault-v1"
    assert len(manifest["solver_module_sha256"]) == 64
    assert manifest["requested_reduction_mm"] == 2.0
    assert manifest["source_geometry_unchanged"] is True
    assert manifest["affected_vertex_count"] > 0
    assert 0.0 < manifest["maximum_actual_vertex_displacement_mm"] <= np.hypot(2.0, 0.2) + 1e-6
    assert (output / "reduction_2.0mm.ply").is_file()
    assert (output / "reduction_2.0mm.glb").is_file()
    assert (output / manifest["output_paths"]["viewer_glb"]).is_file()
    assert (output / "reduction_2.0mm_affected_roi.ply").is_file()
    assert (output / "reduction_2.0mm_moved_vertices.ply").is_file()
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
    import cv2
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
    applied = diagnostics["applied_profile_reduction_mm"]
    assert 3.0 <= applied < 4.0
    assert (
        0.95 * applied
        <= diagnostics["maximum_displacement_mm"]
        <= np.hypot(applied, 0.1 * applied) + 1e-6
    )
    assert diagnostics["median_displacement_mm"] > 0.0
    assert diagnostics["vertices_moved_over_0_1_mm"] > 0
    assert diagnostics["exported_moved_vertex_count"] == manifest["affected_vertex_count"]
    assert (output / manifest["output_paths"]["moved_vertices_ply"]).is_file()
    assert diagnostics["source_mesh_hash"] != diagnostics["output_mesh_hash"]
    assert diagnostics["output_geometry_hash_differs_from_source"] is True
    assert diagnostics["maximum_ply_error_from_memory_mm"] < 1e-6
    assert diagnostics["maximum_profile_change_mm"] >= 3.0
    assert diagnostics["maximum_displacement_distance_from_apex_mm"] < 1.5
    assert (
        diagnostics["maximum_supratip_displacement_mm"]
        < 0.5 * diagnostics["maximum_displacement_mm"]
    )
    assert diagnostics["glb_export"]["geometry_differs_from_source"] is True
    assert diagnostics["glb_export"]["maximum_displacement_from_source_mm"] >= 0.95 * applied
    assert diagnostics["glb_export"]["maximum_transverse_displacement_from_source_mm"] > 0.25
    assert diagnostics["glb_export"]["vertices_with_transverse_change_over_0_1_mm"] > 0
    assert diagnostics["glb_export"]["left_sidewall_median_medial_displacement_mm"] > 0.1
    assert diagnostics["glb_export"]["right_sidewall_median_medial_displacement_mm"] > 0.1
    assert diagnostics["glb_export"]["normals_recomputed_from_simulated_geometry"] is True
    assert diagnostics["glb_export"]["maximum_normal_error_degrees"] < 6.0
    assert diagnostics["glb_export"]["p95_normal_error_degrees"] < 1.0
    assert diagnostics["ply_normals"]["present"] is True
    assert diagnostics["ply_normals"]["recomputed_from_simulated_geometry"] is True
    assert diagnostics["ply_normals"]["maximum_error_degrees"] < 0.01
    assert diagnostics["acceptance"]["passed"] is True
    assert diagnostics["acceptance"]["profile"]["superior_hump_not_left_behind"] is True
    assert diagnostics["acceptance"]["profile"]["smooth_target_followed"] is True
    assert diagnostics["acceptance"]["profile"]["lower_dorsum_not_overcorrected"] is True
    assert diagnostics["acceptance"]["frontal_vault"]["no_central_dent"] is True
    assert diagnostics["acceptance"]["frontal_vault"]["sidewalls_not_moved_outward"] is True
    assert diagnostics["acceptance"]["frontal_vault"]["no_center_strip_pattern"] is True

    simulated = o3d.io.read_triangle_mesh(str(output / manifest["output_paths"]["ply"]))
    simulated_vertices = np.asarray(simulated.vertices)
    assert float(np.max(np.linalg.norm(simulated_vertices - source_vertices, axis=1))) >= (
        0.95 * applied / 1000.0
    )

    roi_mesh = o3d.io.read_triangle_mesh(str(output / manifest["output_paths"]["affected_roi_ply"]))
    roi_vertices_mm = np.asarray(roi_mesh.vertices) * 1000.0
    assert 0 < len(roi_vertices_mm) < len(source_vertices)
    assert np.max(np.abs(roi_vertices_mm[:, 0])) <= (
        manifest["affected_nasal_roi"]["lateral_half_width_mm"] + 0.1
    )
    assert np.max(roi_vertices_mm[:, 1]) < 42.0

    profile_svg = (output / manifest["output_paths"]["profile_comparison_svg"]).read_text()
    assert "red: original" in profile_svg
    assert "gray: target" in profile_svg
    assert "blue: final" in profile_svg
    profile_curve = json.loads(
        (output / manifest["output_paths"]["profile_curve_json"]).read_text()
    )
    assert len(profile_curve["source_anterior_mm"]) == 64
    assert len(profile_curve["target_anterior_mm"]) == 64
    assert len(profile_curve["final_anterior_mm"]) == 64
    assert profile_curve["detected_hump_apex"] == diagnostics["detected_hump_apex"]
    assert set(profile_curve["anatomical_displacements"]) == {
        "radix_nasion",
        "upper_dorsum",
        "hump_apex",
        "mid_dorsum",
        "lower_dorsum",
        "supratip",
        "pronasale",
    }
    assert profile_curve["anatomical_displacements"]["pronasale"][
        "final_posterior_displacement_mm"
    ] == 0.0
    cross_sections = json.loads(
        (output / manifest["output_paths"]["cross_sections_json"]).read_text()
    )
    assert set(cross_sections["sections"]) == {
        "upper_dorsum",
        "hump_region",
        "mid_dorsum",
        "supratip",
    }
    for name, section in cross_sections["sections"].items():
        assert {
            "width_before_mm",
            "width_after_mm",
            "central_height_before_mm",
            "central_height_after_mm",
            "left_sidewall_position_before",
            "left_sidewall_position_after",
            "right_sidewall_position_before",
            "right_sidewall_position_after",
            "left_sidewall_displacement_mm",
            "right_sidewall_displacement_mm",
        } <= set(section)
        assert len(section["source_lateral_mm"]) == 49
        assert len(section["target_lateral_mm"]) == 49
        assert len(section["simulated_lateral_mm"]) == 49
        assert section["final_has_central_dent"] is False
        if name in {"hump_region", "mid_dorsum"}:
            assert section["width_before_mm"] - section["width_after_mm"] > 0.25
            assert section["central_height_before_mm"] > section["central_height_after_mm"]
            assert section["left_sidewall_position_after"]["lateral_mm"] > (
                section["left_sidewall_position_before"]["lateral_mm"]
            )
            assert section["right_sidewall_position_after"]["lateral_mm"] < (
                section["right_sidewall_position_before"]["lateral_mm"]
            )
            assert section["left_sidewall_displacement_mm"] > 0.1
            assert section["right_sidewall_displacement_mm"] > 0.1
            width_ratio = (
                section["simulated_width_at_1_5mm_depth_mm"]
                / section["source_width_at_1_5mm_depth_mm"]
            )
            assert 0.7 <= width_ratio <= 1.05
            assert section["simulated_central_quadratic_curvature_per_mm"] < -0.01
    assert (output / manifest["output_paths"]["cross_sections_svg"]).is_file()
    diagnostic_images = {}
    for key in (
        "front_before_png",
        "front_after_png",
        "profile_before_png",
        "profile_after_png",
        "textured_front_before_png",
        "textured_front_after_png",
        "textured_profile_before_png",
        "textured_profile_after_png",
        "affected_roi_render_png",
        "clay_before_png",
        "clay_after_png",
        "profile_clay_before_png",
        "profile_clay_after_png",
        "top_down_before_png",
        "top_down_after_png",
        "moved_vertices_heatmap_png",
        "front_displacement_heatmap_png",
        "profile_displacement_heatmap_png",
        "normal_before_png",
        "normal_after_png",
    ):
        diagnostic_images[key] = cv2.imread(str(output / manifest["output_paths"][key]))
        assert diagnostic_images[key] is not None
    assert not np.array_equal(
        diagnostic_images["profile_before_png"],
        diagnostic_images["profile_after_png"],
    )
    roi_render = diagnostic_images["affected_roi_render_png"]
    red_highlight = (roi_render[:, :, 2] > 180) & (roi_render[:, :, 1] < 80)
    assert np.count_nonzero(red_highlight) > 0
    viewer_glb = output / manifest["output_paths"]["viewer_glb"]
    assert viewer_glb.is_file()
    assert manifest["exact_final_glb_path"] == str(viewer_glb.resolve())
    assert manifest["output_paths"]["exact_final_glb"] == str(viewer_glb.resolve())
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
    assert manifest["diagnostics"]["acceptance"]["passed"] is True
    assert manifest["diagnostics"]["acceptance"]["identity_request"] is True
    assert (output / manifest["output_paths"]["affected_roi_ply"]).is_file()
