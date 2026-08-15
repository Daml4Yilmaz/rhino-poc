"""Within-subject repeatability metrics for independently reconstructed cases."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
_RANK = {PASS: 0, WARN: 1, FAIL: 2}

MEASUREMENT_REPEATABILITY_LIMITS = {
    "nasofrontal_angle_deg": (3.0, 5.0, "deg"),
    "nasolabial_angle_deg": (3.0, 5.0, "deg"),
    "goode_ratio": (0.03, 0.05, "ratio"),
    "nose_length_mm": (1.0, 2.0, "mm"),
    "nose_width_mm": (1.0, 2.0, "mm"),
    "midline_deviation_mm": (1.0, 2.0, "mm"),
}


def _status(value: float, pass_maximum: float, fail_above: float) -> str:
    return PASS if value <= pass_maximum else WARN if value <= fail_above else FAIL


def _worst(items: list[dict[str, Any]]) -> str:
    return max((item["status"] for item in items), key=_RANK.get) if items else WARN


def measurement_repeatability(case_directories: list[Path]) -> dict[str, Any]:
    """Summarize variation in measurement JSON files for one repeated subject."""
    if len(case_directories) < 2:
        raise ValueError("Repeatability analysis requires at least two independent cases")
    labels: list[str] = []
    documents: list[dict[str, Any]] = []
    for directory in case_directories:
        path = directory / "measurements.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing measurements: {path}")
        labels.append(directory.parent.name if directory.name == "final" else directory.name)
        documents.append(json.loads(path.read_text(encoding="utf-8")))
    measurement_names = list(MEASUREMENT_REPEATABILITY_LIMITS)
    missing = [
        f"{label}:{name}"
        for label, document in zip(labels, documents, strict=True)
        for name in measurement_names
        if name not in document.get("measurements", {})
    ]
    if missing:
        raise ValueError(f"Cases do not share the required measurements: {missing}")

    statistics: dict[str, dict[str, Any]] = {}
    checks: list[dict[str, Any]] = []
    for name in measurement_names:
        values = np.asarray(
            [document["measurements"][name] for document in documents], dtype=np.float64
        )
        pass_maximum, fail_above, unit = MEASUREMENT_REPEATABILITY_LIMITS[name]
        value_range = float(np.ptp(values))
        status = _status(value_range, pass_maximum, fail_above)
        mean = float(np.mean(values))
        standard_deviation = float(np.std(values, ddof=1))
        coefficient_of_variation = (
            abs(standard_deviation / mean) * 100.0 if abs(mean) > 1e-12 else None
        )
        statistics[name] = {
            "values": {label: float(value) for label, value in zip(labels, values, strict=True)},
            "mean": round(mean, 4),
            "sample_standard_deviation": round(standard_deviation, 4),
            "coefficient_of_variation_percent": (
                round(coefficient_of_variation, 3) if coefficient_of_variation is not None else None
            ),
            "maximum_pairwise_difference": round(value_range, 4),
            "unit": unit,
            "status": status,
        }
        checks.append(
            {
                "id": f"repeatability.{name}",
                "label": name,
                "status": status,
                "value": round(value_range, 4),
                "unit": unit,
                "criteria": (
                    f"PASS <= {pass_maximum:g}; WARN <= {fail_above:g}; "
                    f"FAIL > {fail_above:g} maximum pairwise difference"
                ),
                "message": "Candidate engineering limit; surgeon and study approval are required.",
            }
        )
    return {
        "id": "measurement_repeatability",
        "label": "Measurement repeatability",
        "status": _worst(checks),
        "checks": checks,
        "metrics": statistics,
    }


def _rigid_landmark_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:
        right[-1] *= -1
        rotation = right.T @ left.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def _similarity_scale(source: np.ndarray, target: np.ndarray) -> float:
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    covariance = target_centered.T @ source_centered / len(source)
    left, singular_values, right = np.linalg.svd(covariance)
    correction = np.ones(3)
    if np.linalg.det(left) * np.linalg.det(right) < 0:
        correction[-1] = -1
    variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    return float(np.sum(singular_values * correction) / variance)


def _load_surface_case(directory: Path) -> tuple[Any, dict[str, np.ndarray]]:
    import trimesh

    mesh_path = directory / "face_geometry.ply"
    landmark_path = directory / "landmarks.json"
    if not mesh_path.exists() or not landmark_path.exists():
        raise FileNotFoundError(f"Surface repeatability requires mesh and landmarks in {directory}")
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    document = json.loads(landmark_path.read_text(encoding="utf-8"))
    landmarks = {
        name: np.asarray(position, dtype=np.float64) / 1000.0
        for name, position in document["landmarks"].items()
    }
    return mesh, landmarks


def _central_points(vertices: np.ndarray, landmarks: dict[str, np.ndarray]) -> np.ndarray:
    center = np.mean(np.asarray(list(landmarks.values())), axis=0)
    distances = np.linalg.norm(vertices - center, axis=1)
    selected = vertices[distances <= 0.075]
    return selected if len(selected) >= 1000 else vertices


def _nasal_points(vertices: np.ndarray, landmarks: dict[str, np.ndarray]) -> np.ndarray:
    names = ("nasion", "pronasale", "subnasale", "left_alare", "right_alare")
    anchors = np.asarray([landmarks[name] for name in names])
    distances = np.linalg.norm(vertices[:, None, :] - anchors[None, :, :], axis=2)
    selected = vertices[np.min(distances, axis=1) <= 0.022]
    if len(selected) < 500:
        raise RuntimeError("Fewer than 500 vertices were found in the automatic nasal region")
    return selected


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _surface_distances(points: np.ndarray, target_mesh: Any) -> np.ndarray:
    import open3d as o3d

    tensor_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(target_mesh.vertices), dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(np.asarray(target_mesh.faces), dtype=o3d.core.Dtype.Int64),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    return scene.compute_distance(
        o3d.core.Tensor(points.astype(np.float32), dtype=o3d.core.Dtype.Float32)
    ).numpy()


def pairwise_surface_repeatability(first: Path, second: Path) -> dict[str, Any]:
    """Rigidly align central facial surfaces and compare the nasal regions."""
    import open3d as o3d
    import trimesh

    source_mesh, source_landmarks = _load_surface_case(first)
    target_mesh, target_landmarks = _load_surface_case(second)
    common = sorted(source_landmarks.keys() & target_landmarks.keys())
    if len(common) < 6:
        raise RuntimeError("At least six common landmarks are required to initialize alignment")
    source_array = np.asarray([source_landmarks[name] for name in common])
    target_array = np.asarray([target_landmarks[name] for name in common])
    initial = _rigid_landmark_transform(source_array, target_array)

    source_central = _central_points(np.asarray(source_mesh.vertices), source_landmarks)
    target_central = _central_points(np.asarray(target_mesh.vertices), target_landmarks)
    source_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_central))
    target_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_central))
    source_cloud = source_cloud.voxel_down_sample(0.0015)
    target_cloud = target_cloud.voxel_down_sample(0.0015)
    registration = o3d.pipelines.registration.registration_icp(
        source_cloud,
        target_cloud,
        0.006,
        initial,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    transform = np.asarray(registration.transformation)
    transformed_source_vertices = _transform_points(np.asarray(source_mesh.vertices), transform)
    transformed_source_mesh = trimesh.Trimesh(
        vertices=transformed_source_vertices,
        faces=np.asarray(source_mesh.faces),
        process=False,
    )
    transformed_source_landmarks = {
        name: _transform_points(point[None, :], transform)[0]
        for name, point in source_landmarks.items()
    }
    source_nose = _nasal_points(transformed_source_vertices, transformed_source_landmarks)
    target_nose = _nasal_points(np.asarray(target_mesh.vertices), target_landmarks)
    distances_mm = (
        np.concatenate(
            [
                _surface_distances(source_nose, target_mesh),
                _surface_distances(target_nose, transformed_source_mesh),
            ]
        )
        * 1000.0
    )
    landmark_scale = _similarity_scale(source_array, target_array)
    return {
        "first_case": first.parent.name if first.name == "final" else first.name,
        "second_case": second.parent.name if second.name == "final" else second.name,
        "alignment_method": "rigid_landmark_initialization_then_central_face_icp_v1",
        "scale_was_optimized": False,
        "diagnostic_landmark_similarity_scale": round(landmark_scale, 6),
        "icp_fitness": round(float(registration.fitness), 6),
        "icp_inlier_rmse_mm": round(float(registration.inlier_rmse) * 1000.0, 4),
        "symmetric_nasal_sample_count": len(distances_mm),
        "symmetric_nasal_distance_median_mm": round(float(np.median(distances_mm)), 4),
        "symmetric_nasal_distance_p95_mm": round(float(np.percentile(distances_mm, 95)), 4),
        "symmetric_nasal_distance_p99_mm": round(float(np.percentile(distances_mm, 99)), 4),
    }


def surface_repeatability(case_directories: list[Path]) -> dict[str, Any]:
    pairs = [
        pairwise_surface_repeatability(case_directories[first], case_directories[second])
        for first in range(len(case_directories))
        for second in range(first + 1, len(case_directories))
    ]
    maximum_p95 = max(pair["symmetric_nasal_distance_p95_mm"] for pair in pairs)
    status = _status(maximum_p95, 1.0, 2.0)
    check = {
        "id": "repeatability.nasal_surface_p95",
        "label": "Worst pairwise symmetric nasal-surface p95 distance",
        "status": status,
        "value": round(maximum_p95, 4),
        "unit": "mm",
        "criteria": "PASS <= 1; WARN <= 2; FAIL > 2 after rigid alignment without scale",
        "message": "Repeatability metric, not absolute accuracy against ground truth.",
    }
    return {
        "id": "surface_repeatability",
        "label": "Nasal surface repeatability",
        "status": status,
        "checks": [check],
        "metrics": {"pairwise_comparisons": pairs},
    }


def build_repeatability_report(
    case_directories: list[Path],
    *,
    subject_id: str,
    include_surface: bool = True,
) -> dict[str, Any]:
    directories = [path.expanduser().resolve() for path in case_directories]
    sections = [measurement_repeatability(directories)]
    if include_surface:
        sections.append(surface_repeatability(directories))
    return {
        "schema_version": 1,
        "profile_id": "poc_repeatability_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subject_id": subject_id,
        "case_directories": [str(path) for path in directories],
        "overall_status": _worst(sections),
        "clinical_use_authorized": False,
        "warning": (
            "Candidate engineering limits. Repeatability does not establish accuracy, and one "
            "subject cannot estimate population reliability or ICC."
        ),
        "sections": sections,
    }


def write_repeatability_report(
    report: dict[str, Any], json_path: Path, html_path: Path | None = None
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if html_path is not None:
        rows = []
        for section in report["sections"]:
            rows.append(
                f'<tr><th colspan="5">{escape(section["label"])} — {section["status"]}</th></tr>'
            )
            for check in section["checks"]:
                rows.append(
                    f"<tr><td>{check['status']}</td><td>{escape(check['label'])}</td>"
                    f"<td>{check['value']} {escape(check['unit'])}</td>"
                    f"<td>{escape(check['criteria'])}</td><td>{escape(check['message'])}</td></tr>"
                )
        html_path.write_text(
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            "<title>Repeatability report</title><style>body{font:15px system-ui;margin:2rem}"
            "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:.55rem;"
            "text-align:left}</style></head><body><h1>Within-subject repeatability report</h1>"
            f"<p><strong>Overall: {report['overall_status']}</strong></p>"
            f"<p>{escape(report['warning'])}</p><table>{''.join(rows)}</table></body></html>",
            encoding="utf-8",
        )
