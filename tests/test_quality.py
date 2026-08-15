import json
from pathlib import Path

from poc.quality import (
    FAIL,
    PASS,
    WARN,
    _dense_fusion_section,
    build_capture_report,
    evaluate_capture,
)
from poc.report.repeatability import measurement_repeatability


def _good_capture() -> dict:
    return {
        "frame_count": 1200,
        "rgb_size": [1920, 1440],
        "duration_seconds": 32.0,
        "effective_fps": 37.5,
        "path_length_m": 1.5,
        "trajectory_span_m": 0.5,
        "orientation_coverage_degrees": 145.0,
        "tracking_jump_count": 0,
        "linear_speed_p95_m_per_s": 0.1,
        "angular_speed_p95_deg_per_s": 20.0,
        "focal_length_drift_percent": 0.5,
        "has_depth": True,
        "depth_frame_coverage": 1.0,
    }


def _good_video() -> dict:
    return {
        "selected_sharpness_p10": 120.0,
        "selected_sharpness_median": 250.0,
        "median_dark_pixel_percent": 0.1,
        "median_bright_pixel_percent": 0.1,
        "luminance_temporal_range": 20.0,
    }


def test_capture_quality_preserves_worst_gate() -> None:
    section = evaluate_capture(_good_capture(), _good_video(), selected_count=120)
    assert section["status"] == PASS

    warning_capture = _good_capture()
    warning_capture["duration_seconds"] = 20.0
    assert evaluate_capture(warning_capture, _good_video(), 120)["status"] == WARN

    failed_capture = _good_capture()
    failed_capture["orientation_coverage_degrees"] = 90.0
    report = build_capture_report(failed_capture, _good_video(), 120)
    assert report["overall_status"] == FAIL
    assert report["clinical_use_authorized"] is False


def _write_measurements(directory: Path, offset: float) -> None:
    directory.mkdir()
    values = {
        "nasofrontal_angle_deg": 160.0 + offset,
        "nasolabial_angle_deg": 95.0 + offset,
        "goode_ratio": 0.50 + offset / 100.0,
        "nose_length_mm": 45.0 + offset,
        "nose_width_mm": 30.0 + offset,
        "midline_deviation_mm": 1.0 + offset,
    }
    (directory / "measurements.json").write_text(
        json.dumps({"measurements": values}), encoding="utf-8"
    )


def test_measurement_repeatability_reports_pairwise_range(tmp_path: Path) -> None:
    first = tmp_path / "case_1"
    second = tmp_path / "case_2"
    third = tmp_path / "case_3"
    _write_measurements(first, 0.0)
    _write_measurements(second, 0.5)
    _write_measurements(third, 2.5)

    section = measurement_repeatability([first, second, third])
    assert section["status"] == FAIL
    assert section["metrics"]["nose_length_mm"]["maximum_pairwise_difference"] == 2.5
    assert section["metrics"]["nose_width_mm"]["sample_standard_deviation"] > 0


def test_dense_fusion_gate_requires_the_recorded_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "face_dense_fused.ply"
    artifact.write_bytes(b"unaltered fusion output")
    document = {
        "role": "raw_fused_dense_point_cloud_pre_poisson",
        "generated_by": "COLMAP stereo_fusion",
        "path": artifact.name,
        "point_count": 150_000,
        "nonfinite_point_count": 0,
        "file_size_bytes": artifact.stat().st_size,
        "bounding_box_extent": [1.0, 2.0, 3.0],
        "normals_present_in_persisted_artifact": True,
    }

    assert _dense_fusion_section(tmp_path, document)["status"] == PASS
    artifact.unlink()
    assert _dense_fusion_section(tmp_path, document)["status"] == FAIL
