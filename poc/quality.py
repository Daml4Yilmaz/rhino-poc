"""Versioned, auditable acceptance criteria for reconstruction cases.

The thresholds in this module are engineering gates for the proof of concept.
They are deliberately not described as clinical validation limits.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

PROFILE_ID = "poc_engineering_v1"
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
_STATUS_RANK = {PASS: 0, WARN: 1, FAIL: 2}


def _worst_status(items: list[dict[str, Any]]) -> str:
    statuses = [item["status"] for item in items]
    return max(statuses, key=_STATUS_RANK.get) if statuses else WARN


def _check(
    identifier: str,
    label: str,
    status: str,
    value: Any,
    unit: str,
    criteria: str,
    message: str,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "label": label,
        "status": status,
        "value": value,
        "unit": unit,
        "criteria": criteria,
        "message": message,
    }


def _minimum(
    identifier: str,
    label: str,
    value: float | None,
    pass_minimum: float,
    fail_below: float,
    unit: str,
) -> dict[str, Any]:
    criteria = f"PASS >= {pass_minimum:g}; WARN >= {fail_below:g}; FAIL < {fail_below:g}"
    if value is None:
        return _check(identifier, label, WARN, None, unit, criteria, "Metric was not available.")
    status = PASS if value >= pass_minimum else WARN if value >= fail_below else FAIL
    return _check(identifier, label, status, value, unit, criteria, "Threshold evaluated.")


def _maximum(
    identifier: str,
    label: str,
    value: float | None,
    pass_maximum: float,
    fail_above: float,
    unit: str,
) -> dict[str, Any]:
    criteria = f"PASS <= {pass_maximum:g}; WARN <= {fail_above:g}; FAIL > {fail_above:g}"
    if value is None:
        return _check(identifier, label, WARN, None, unit, criteria, "Metric was not available.")
    status = PASS if value <= pass_maximum else WARN if value <= fail_above else FAIL
    return _check(identifier, label, status, value, unit, criteria, "Threshold evaluated.")


def _range(
    identifier: str,
    label: str,
    value: float | None,
    pass_range: tuple[float, float],
    warn_range: tuple[float, float],
    unit: str,
) -> dict[str, Any]:
    criteria = (
        f"PASS {pass_range[0]:g}-{pass_range[1]:g}; "
        f"WARN {warn_range[0]:g}-{warn_range[1]:g}; FAIL outside WARN range"
    )
    if value is None:
        return _check(identifier, label, WARN, None, unit, criteria, "Metric was not available.")
    status = PASS if pass_range[0] <= value <= pass_range[1] else WARN
    if value < warn_range[0] or value > warn_range[1]:
        status = FAIL
    return _check(identifier, label, status, value, unit, criteria, "Threshold evaluated.")


def _section(identifier: str, label: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": identifier,
        "label": label,
        "status": _worst_status(checks),
        "checks": checks,
    }


def evaluate_capture(
    capture: dict[str, Any],
    video_quality: dict[str, Any] | None = None,
    selected_count: int | None = None,
) -> dict[str, Any]:
    """Evaluate a synchronized Stray capture against the PoC input protocol."""
    video_quality = video_quality or {}
    width, height = (capture.get("rgb_size") or [None, None])[:2]
    longest_side = max(value for value in (width, height) if value is not None) if width else None
    checks = [
        _minimum(
            "capture.frame_count",
            "Synchronized frames",
            capture.get("frame_count"),
            750,
            540,
            "frames",
        ),
        _range(
            "capture.duration",
            "Capture duration",
            capture.get("duration_seconds"),
            (25, 45),
            (18, 60),
            "s",
        ),
        _minimum(
            "capture.fps", "Effective frame rate", capture.get("effective_fps"), 29, 24, "fps"
        ),
        _minimum("capture.resolution", "Longest RGB dimension", longest_side, 1920, 1280, "px"),
        _minimum(
            "capture.path", "Camera path length", capture.get("path_length_m"), 1.2, 0.75, "m"
        ),
        _minimum(
            "capture.span", "Trajectory span", capture.get("trajectory_span_m"), 0.4, 0.25, "m"
        ),
        _minimum(
            "capture.orientation_coverage",
            "View-direction coverage",
            capture.get("orientation_coverage_degrees"),
            140,
            120,
            "deg",
        ),
        _maximum(
            "capture.tracking_jumps",
            "Tracking jumps over 15 cm",
            capture.get("tracking_jump_count"),
            0,
            1,
            "frames",
        ),
        _maximum(
            "capture.focal_drift",
            "Focal-length drift",
            capture.get("focal_length_drift_percent"),
            1,
            2,
            "%",
        ),
        _maximum(
            "capture.linear_speed",
            "95th-percentile camera speed",
            capture.get("linear_speed_p95_m_per_s"),
            0.15,
            0.30,
            "m/s",
        ),
        _maximum(
            "capture.angular_speed",
            "95th-percentile angular speed",
            capture.get("angular_speed_p95_deg_per_s"),
            30,
            60,
            "deg/s",
        ),
        _minimum(
            "video.sharpness_p10",
            "Evaluated-frame sharpness p10",
            video_quality.get("selected_sharpness_p10"),
            80,
            40,
            "Laplacian variance",
        ),
        _minimum(
            "video.sharpness_median",
            "Evaluated-frame median sharpness",
            video_quality.get("selected_sharpness_median"),
            150,
            80,
            "Laplacian variance",
        ),
        _maximum(
            "video.dark_pixels",
            "Median clipped-black pixels",
            video_quality.get("median_dark_pixel_percent"),
            1,
            5,
            "%",
        ),
        _maximum(
            "video.bright_pixels",
            "Median clipped-white pixels",
            video_quality.get("median_bright_pixel_percent"),
            1,
            5,
            "%",
        ),
        _maximum(
            "video.luminance_stability",
            "Temporal luminance range",
            video_quality.get("luminance_temporal_range"),
            40,
            70,
            "8-bit levels",
        ),
    ]
    if selected_count is not None:
        checks.append(
            _minimum(
                "capture.selected_frames",
                "Selected reconstruction frames",
                selected_count,
                100,
                60,
                "frames",
            )
        )
    if capture.get("has_depth"):
        checks.append(
            _minimum(
                "capture.depth_coverage",
                "LiDAR depth-frame coverage",
                capture.get("depth_frame_coverage"),
                0.95,
                0.80,
                "ratio",
            )
        )
    else:
        checks.append(
            _check(
                "capture.depth_coverage",
                "LiDAR depth-frame coverage",
                WARN,
                0,
                "ratio",
                "PASS >= 0.95 when LiDAR is available; otherwise WARN",
                "No depth stream is available; metric scale must rely on the ARKit trajectory.",
            )
        )
    return _section("input_capture", "Input capture", checks)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def analyze_mesh(mesh_path: Path) -> dict[str, Any]:
    """Calculate topology and sampling metrics without modifying the mesh."""
    import trimesh

    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if not len(vertices) or not len(faces):
        raise RuntimeError(f"Mesh is empty: {mesh_path}")
    triangles = vertices[faces]
    double_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    degenerate_count = int(np.count_nonzero(double_areas <= 2e-14))
    edge_lengths = np.linalg.norm(
        np.concatenate(
            [
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 1],
                triangles[:, 0] - triangles[:, 2],
            ],
            axis=0,
        ),
        axis=1,
    )
    component_labels = trimesh.graph.connected_component_labels(
        loaded.face_adjacency, node_count=len(faces)
    )
    component_faces = np.bincount(component_labels)
    largest_ratio = float(component_faces.max()) / len(faces) if len(component_faces) else 0.0
    edge_counts = np.bincount(loaded.edges_unique_inverse)
    return {
        "vertex_count": len(vertices),
        "triangle_count": len(faces),
        "nonfinite_vertex_count": int(np.size(vertices) - np.count_nonzero(np.isfinite(vertices))),
        "degenerate_triangle_count": degenerate_count,
        "connected_component_count": len(component_faces),
        "largest_component_ratio": round(float(largest_ratio), 6),
        "boundary_edge_count": int(np.count_nonzero(edge_counts == 1)),
        "is_watertight": bool(loaded.is_watertight),
        "is_winding_consistent": bool(loaded.is_winding_consistent),
        "median_edge_length_mm": round(float(np.median(edge_lengths)) * 1000.0, 4),
        "edge_length_p95_mm": round(float(np.percentile(edge_lengths, 95)) * 1000.0, 4),
    }


def _mesh_section(mesh_path: Path) -> dict[str, Any]:
    if not mesh_path.exists():
        return _section(
            "authoritative_mesh",
            "Authoritative mesh",
            [
                _check(
                    "mesh.exists",
                    "Mesh exists",
                    FAIL,
                    False,
                    "",
                    "PASS when present",
                    "Authoritative mesh is missing.",
                )
            ],
        )
    metrics = analyze_mesh(mesh_path)
    checks = [
        _maximum(
            "mesh.nonfinite",
            "Non-finite vertex values",
            metrics["nonfinite_vertex_count"],
            0,
            0,
            "values",
        ),
        _maximum(
            "mesh.degenerate",
            "Degenerate triangles",
            metrics["degenerate_triangle_count"],
            0,
            50,
            "triangles",
        ),
        _minimum(
            "mesh.largest_component",
            "Largest connected component",
            metrics["largest_component_ratio"],
            0.999,
            0.99,
            "ratio",
        ),
        _maximum(
            "mesh.component_count",
            "Connected components",
            metrics["connected_component_count"],
            1,
            50,
            "components",
        ),
        _range(
            "mesh.edge_sampling",
            "Median edge length",
            metrics["median_edge_length_mm"],
            (0.2, 1.0),
            (0.1, 1.5),
            "mm",
        ),
        _check(
            "mesh.winding",
            "Consistent triangle winding",
            PASS if metrics["is_winding_consistent"] else FAIL,
            metrics["is_winding_consistent"],
            "",
            "PASS when true",
            "Triangle orientation must be consistent for rendering and raycasting.",
        ),
        _check(
            "mesh.watertight",
            "Watertight surface",
            PASS if metrics["is_watertight"] else WARN,
            metrics["is_watertight"],
            "",
            "PASS when true; WARN is allowed for an explicitly cropped facial surface",
            "An open boundary is expected only at the documented face crop.",
        ),
    ]
    section = _section("authoritative_mesh", "Authoritative mesh", checks)
    section["metrics"] = metrics
    return section


def _landmark_section(document: dict[str, Any]) -> dict[str, Any]:
    quality = document.get("quality", {})
    if not quality:
        return _section(
            "landmarks",
            "Landmark registration",
            [
                _check(
                    "landmarks.quality",
                    "Landmark quality metadata",
                    FAIL,
                    None,
                    "",
                    "PASS when present",
                    "No landmark-quality metadata was found.",
                )
            ],
        )
    inlier_ratios = [entry["inliers"] / entry["observations"] for entry in quality.values()]
    residuals = [entry.get("median_ray_residual_mm", float("inf")) for entry in quality.values()]
    snaps = [entry.get("surface_snap_distance_mm", float("inf")) for entry in quality.values()]
    checks = [
        _minimum(
            "landmarks.minimum_inlier_ratio",
            "Minimum multi-view inlier ratio",
            round(min(inlier_ratios), 4),
            0.70,
            0.50,
            "ratio",
        ),
        _maximum(
            "landmarks.maximum_ray_residual",
            "Maximum median ray residual",
            round(max(residuals), 3),
            1.5,
            2.0,
            "mm",
        ),
        _maximum(
            "landmarks.maximum_surface_snap",
            "Maximum surface-snap distance",
            round(max(snaps), 3),
            1.0,
            2.0,
            "mm",
        ),
    ]
    surface_hit_counts = [entry.get("surface_hit_count") for entry in quality.values()]
    surface_p95 = [entry.get("surface_dispersion_p95_mm") for entry in quality.values()]
    if all(value is not None for value in surface_hit_counts):
        checks.append(
            _minimum(
                "landmarks.minimum_surface_hits",
                "Minimum visible surface intersections",
                min(surface_hit_counts),
                20,
                10,
                "views",
            )
        )
    if all(value is not None for value in surface_p95):
        checks.append(
            _maximum(
                "landmarks.maximum_surface_dispersion_p95",
                "Maximum surface-consensus p95 dispersion",
                round(max(surface_p95), 3),
                2.0,
                4.0,
                "mm",
            )
        )
    section = _section("landmarks", "Landmark registration", checks)
    section["metrics"] = {
        "landmark_count": len(quality),
        "minimum_inlier_ratio": round(min(inlier_ratios), 4),
        "maximum_median_ray_residual_mm": round(max(residuals), 3),
        "maximum_surface_snap_distance_mm": round(max(snaps), 3),
        "worst_surface_snap_landmark": max(
            quality, key=lambda name: quality[name].get("surface_snap_distance_mm", float("inf"))
        ),
    }
    return section


def _scale_section(document: dict[str, Any]) -> dict[str, Any]:
    if not document:
        return _section(
            "metric_scale",
            "Metric scale",
            [
                _check(
                    "scale.exists",
                    "Scale report exists",
                    FAIL,
                    False,
                    "",
                    "PASS when present",
                    "scale.json is missing.",
                )
            ],
        )
    verified = document.get("scale_verified")
    verified_status = PASS if verified is True else FAIL if verified is False else WARN
    checks = [
        _minimum(
            "scale.pose_inlier_ratio",
            "ARKit/COLMAP pose inlier ratio",
            document.get("pose_inlier_ratio"),
            0.80,
            0.60,
            "ratio",
        ),
        _maximum(
            "scale.pose_median_residual",
            "Pose alignment median residual",
            document.get("pose_residual_median_mm"),
            10,
            20,
            "mm",
        ),
        _maximum(
            "scale.pose_p95_residual",
            "Pose alignment p95 residual",
            document.get("pose_residual_p95_mm"),
            20,
            30,
            "mm",
        ),
        _check(
            "scale.lidar_verification",
            "Independent LiDAR scale verification",
            verified_status,
            verified,
            "",
            "PASS when verified; WARN when unavailable; FAIL when sources disagree",
            "LiDAR verification is independent of the ARKit trajectory scale.",
        ),
    ]
    return _section("metric_scale", "Metric scale", checks)


def _sfm_section(document: dict[str, Any]) -> dict[str, Any]:
    if not document:
        return _section(
            "sparse_reconstruction",
            "Sparse reconstruction",
            [
                _check(
                    "sfm.metrics",
                    "Sparse metrics available",
                    WARN,
                    None,
                    "",
                    "PASS when sfm.json is present",
                    "Legacy compact bundles did not retain structured sparse metrics.",
                )
            ],
        )
    checks = [
        _minimum(
            "sfm.registered_images",
            "Registered images",
            document.get("registered_image_count"),
            80,
            40,
            "images",
        ),
        _minimum(
            "sfm.registration_ratio",
            "Image registration ratio",
            document.get("registration_ratio"),
            0.85,
            0.60,
            "ratio",
        ),
        _minimum(
            "sfm.sparse_points",
            "Sparse 3D points",
            document.get("sparse_point_count"),
            10_000,
            2_000,
            "points",
        ),
    ]
    return _section("sparse_reconstruction", "Sparse reconstruction", checks)


def _asset_section(
    case_dir: Path,
    geometry: dict[str, Any],
    landmarks: dict[str, Any],
    measurements: dict[str, Any],
) -> dict[str, Any]:
    geometry_ids = [
        item.get("geometry_id")
        for item in (geometry, landmarks, measurements)
        if item.get("geometry_id")
    ]
    identity_matches = bool(geometry_ids) and len(set(geometry_ids)) == 1 and len(geometry_ids) == 3
    render = geometry.get("render_asset", {})
    maximum_deviation = render.get("maximum_position_deviation_m")
    texture_path = case_dir / "texture" / "texture.png"
    textured_mesh_path = case_dir / "texture" / "mesh.ply"
    texture_size: list[int] | None = None
    sampled_dark_percent: float | None = None
    sampled_bright_percent: float | None = None
    sampled_median_luminance: float | None = None
    if texture_path.exists():
        import cv2

        texture = cv2.imread(str(texture_path), cv2.IMREAD_UNCHANGED)
        if texture is not None:
            texture_size = [int(texture.shape[1]), int(texture.shape[0])]
            if textured_mesh_path.exists():
                import trimesh

                textured_mesh = trimesh.load(textured_mesh_path, force="mesh", process=False)
                uv_values = getattr(textured_mesh.visual, "uv", None)
                uv = (
                    np.asarray(uv_values, dtype=np.float64)
                    if uv_values is not None
                    else np.empty((0, 2), dtype=np.float64)
                )
                faces = np.asarray(textured_mesh.faces, dtype=np.int64)
                if len(uv) and len(faces):
                    face_uv = uv[faces].mean(axis=1)
                    height, width = texture.shape[:2]
                    columns = np.clip(
                        np.rint(face_uv[:, 0] * (width - 1)).astype(np.int64), 0, width - 1
                    )
                    rows = np.clip(
                        np.rint((1.0 - face_uv[:, 1]) * (height - 1)).astype(np.int64),
                        0,
                        height - 1,
                    )
                    sampled = texture[rows, columns]
                    if sampled.ndim == 2 and sampled.shape[1] >= 3:
                        gray = cv2.cvtColor(sampled[:, None, :3], cv2.COLOR_BGR2GRAY)[:, 0]
                        sampled_dark_percent = round(float(np.mean(gray <= 5) * 100.0), 4)
                        sampled_bright_percent = round(float(np.mean(gray >= 250) * 100.0), 4)
                        sampled_median_luminance = round(float(np.median(gray)), 3)
    texture_longest = max(texture_size) if texture_size else None
    checks = [
        _check(
            "assets.geometry_identity",
            "Geometry identity across assets",
            PASS if identity_matches else FAIL,
            identity_matches,
            "",
            "PASS when geometry, landmarks, and measurements share one ID",
            "All interactive assets must refer to the authoritative geometry.",
        ),
        _maximum(
            "assets.render_deviation",
            "Render-to-authoritative maximum deviation",
            maximum_deviation,
            1e-6,
            1e-4,
            "m",
        ),
        _minimum(
            "assets.texture_resolution",
            "Texture atlas longest dimension",
            texture_longest,
            4096,
            2048,
            "px",
        ),
        _maximum(
            "assets.texture_black_faces",
            "Face-sampled near-black texture",
            sampled_dark_percent,
            2,
            8,
            "%",
        ),
        _maximum(
            "assets.texture_white_faces",
            "Face-sampled near-white texture",
            sampled_bright_percent,
            1,
            5,
            "%",
        ),
        _check(
            "assets.skin_material",
            "Non-metallic PBR skin material",
            PASS
            if render.get("material", {}).get("model") == "pbr_metallic_roughness"
            and render.get("material", {}).get("metallic_factor") == 0.0
            else WARN,
            render.get("material"),
            "",
            "PASS for registered PBR material with metallic factor 0",
            "Legacy assets without explicit skin material are reported as WARN.",
        ),
    ]
    section = _section("registered_assets", "Registered measurement and visual assets", checks)
    section["metrics"] = {
        "geometry_id": geometry.get("geometry_id"),
        "texture_size": texture_size,
        "face_sampled_dark_percent": sampled_dark_percent,
        "face_sampled_bright_percent": sampled_bright_percent,
        "face_sampled_median_luminance": sampled_median_luminance,
    }
    return section


def _pipeline_section(manifest: dict[str, Any]) -> dict[str, Any]:
    stages = manifest.get("stages", {})
    elapsed = sum(
        float(stage.get("metadata", {}).get("elapsed_seconds", 0.0))
        for stage in stages.values()
        if stage.get("status") == "complete"
    )
    incomplete = sorted(
        name
        for name, stage in stages.items()
        if name != "quality" and stage.get("status") != "complete"
    )
    checks = [
        _maximum(
            "pipeline.runtime", "Total recorded stage runtime", round(elapsed, 3), 3600, 5400, "s"
        ),
        _check(
            "pipeline.stage_completion",
            "Recorded stages complete",
            PASS if not incomplete else FAIL,
            incomplete,
            "",
            "PASS when no recorded stage is incomplete",
            "A failed or interrupted stage invalidates the case.",
        ),
    ]
    return _section("pipeline_execution", "Pipeline execution", checks)


def _measurement_section(document: dict[str, Any]) -> dict[str, Any]:
    present = len(document.get("measurements", {})) == 6
    checks = [
        _check(
            "measurements.count",
            "Six measurements present",
            PASS if present else FAIL,
            len(document.get("measurements", {})),
            "measurements",
            "PASS when exactly six are present",
            "The PoC contract requires six outputs.",
        ),
        _check(
            "measurements.definition_validation",
            "Surgeon-approved measurement definitions",
            WARN,
            document.get("definition"),
            "",
            "WARN until definitions and reference planes are approved and validated",
            "Current values are research outputs and must not be used clinically.",
        ),
    ]
    return _section("measurements", "Measurements", checks)


def build_capture_report(
    capture: dict[str, Any],
    video_quality: dict[str, Any] | None = None,
    selected_count: int | None = None,
) -> dict[str, Any]:
    section = evaluate_capture(capture, video_quality, selected_count)
    return {
        "schema_version": 1,
        "profile_id": PROFILE_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "input_capture_preflight",
        "overall_status": section["status"],
        "clinical_use_authorized": False,
        "warning": "Engineering acceptance profile; not a clinically validated device specification.",
        "sections": [section],
    }


def build_case_report(case_dir: Path) -> dict[str, Any]:
    """Evaluate available artifacts for one completed reconstruction case."""
    case_dir = case_dir.expanduser().resolve()
    manifest = _load_json(case_dir / "case.json")
    ingest_metadata = manifest.get("stages", {}).get("ingest", {}).get("metadata", {})
    frame_index = _load_json(case_dir / "frames.json")
    capture_section = evaluate_capture(
        ingest_metadata.get("capture", {}),
        frame_index.get("video_quality", {}),
        ingest_metadata.get("selected_images", frame_index.get("selected_count")),
    )
    mask = manifest.get("stages", {}).get("mask", {}).get("metadata", {})
    mask_section = _section(
        "face_masking",
        "Face masking",
        [
            _minimum(
                "mask.success_ratio",
                "Accepted face-mask ratio",
                mask.get("success_ratio"),
                0.95,
                0.85,
                "ratio",
            )
        ],
    )
    geometry = _load_json(case_dir / "geometry.json")
    landmarks = _load_json(case_dir / "landmarks.json")
    measurements = _load_json(case_dir / "measurements.json")
    sections = [
        capture_section,
        mask_section,
        _sfm_section(_load_json(case_dir / "sfm.json")),
        _scale_section(_load_json(case_dir / "scale.json")),
        _mesh_section(case_dir / "face_geometry.ply"),
        _landmark_section(landmarks),
        _asset_section(case_dir, geometry, landmarks, measurements),
        _measurement_section(measurements),
        _pipeline_section(manifest),
    ]
    overall = _worst_status(sections)
    failed = [section["label"] for section in sections if section["status"] == FAIL]
    warned = [section["label"] for section in sections if section["status"] == WARN]
    return {
        "schema_version": 1,
        "profile_id": PROFILE_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "complete_reconstruction_case",
        "case_directory": str(case_dir),
        "geometry_id": geometry.get("geometry_id"),
        "overall_status": overall,
        "clinical_use_authorized": False,
        "warning": "Engineering acceptance profile; not a clinically validated device specification.",
        "summary": {"failed_sections": failed, "warning_sections": warned},
        "sections": sections,
    }


def write_report(report: dict[str, Any], json_path: Path, html_path: Path | None = None) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if html_path is not None:
        html_path.write_text(render_html(report), encoding="utf-8")


def render_html(report: dict[str, Any]) -> str:
    """Render a dependency-free human-readable companion to the JSON report."""
    colors = {PASS: "#18794e", WARN: "#946200", FAIL: "#c62828"}
    rows: list[str] = []
    for section in report["sections"]:
        rows.append(
            f'<tr class="section"><th colspan="6">{escape(section["label"])} '
            f'<span style="color:{colors[section["status"]]}">{section["status"]}</span></th></tr>'
        )
        for check in section["checks"]:
            value = "not available" if check["value"] is None else str(check["value"])
            rows.append(
                "<tr>"
                f'<td style="color:{colors[check["status"]]}">{check["status"]}</td>'
                f"<td>{escape(check['label'])}</td><td>{escape(value)}</td>"
                f"<td>{escape(check['unit'])}</td><td>{escape(check['criteria'])}</td>"
                f"<td>{escape(check['message'])}</td></tr>"
            )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Rhino PoC quality report</title>
<style>body{{font:15px system-ui;margin:2rem;color:#202124}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #d9dde3;padding:.55rem;text-align:left;vertical-align:top}}
.section th{{background:#f4f6f8;font-size:1.05rem}}code{{background:#f4f6f8;padding:.15rem .3rem}}
</style></head><body><h1>Reconstruction quality report</h1>
<p><strong>Overall: <span style="color:{colors[report["overall_status"]]}">{report["overall_status"]}</span></strong></p>
<p>Profile: <code>{escape(report["profile_id"])}</code>. {escape(report["warning"])}</p>
<table><thead><tr><th>Status</th><th>Check</th><th>Value</th><th>Unit</th><th>Acceptance criteria</th><th>Interpretation</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></body></html>"""
