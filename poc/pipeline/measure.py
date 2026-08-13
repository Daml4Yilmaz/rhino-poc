"""Compute six provisional standard rhinoplasty surface measurements."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from poc.logging_utils import get_logger

REQUIRED_LANDMARKS = {
    "glabella",
    "nasion",
    "pronasale",
    "subnasale",
    "columella",
    "labiale_superius",
    "left_alare",
    "right_alare",
    "left_endocanthion",
    "right_endocanthion",
}


def _angle(first: np.ndarray, vertex: np.ndarray, third: np.ndarray) -> float:
    first_vector = first - vertex
    second_vector = third - vertex
    denominator = np.linalg.norm(first_vector) * np.linalg.norm(second_vector)
    if denominator < 1e-9:
        raise ValueError("Cannot calculate an angle from coincident landmarks")
    cosine = np.dot(first_vector, second_vector) / denominator
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def compute_measurements(landmarks: dict[str, list[float]]) -> dict[str, float]:
    missing = sorted(REQUIRED_LANDMARKS - landmarks.keys())
    if missing:
        raise ValueError(f"Missing required landmarks: {missing}")
    points = {name: np.asarray(value, dtype=np.float64) for name, value in landmarks.items()}

    eye_axis = points["right_endocanthion"] - points["left_endocanthion"]
    eye_axis /= np.linalg.norm(eye_axis)
    eye_midpoint = (points["left_endocanthion"] + points["right_endocanthion"]) / 2.0
    nasal_length = float(np.linalg.norm(points["pronasale"] - points["nasion"]))
    alar_midpoint = (points["left_alare"] + points["right_alare"]) / 2.0
    tip_projection = float(np.linalg.norm(points["pronasale"] - alar_midpoint))
    midline_deviation = float(abs(np.dot(points["pronasale"] - eye_midpoint, eye_axis)))

    return {
        "nasofrontal_angle_deg": round(
            _angle(points["glabella"], points["nasion"], points["pronasale"]), 1
        ),
        "nasolabial_angle_deg": round(
            _angle(points["columella"], points["subnasale"], points["labiale_superius"]),
            1,
        ),
        "goode_ratio": round(tip_projection / nasal_length, 3),
        "nose_length_mm": round(nasal_length, 2),
        "nose_width_mm": round(
            float(np.linalg.norm(points["left_alare"] - points["right_alare"])), 2
        ),
        "midline_deviation_mm": round(midline_deviation, 2),
    }


def run_measurements(landmarks_json: Path, output_json: Path) -> dict[str, float]:
    document = json.loads(landmarks_json.read_text(encoding="utf-8"))
    landmarks = document.get("landmarks", document)
    measurements = compute_measurements(landmarks)
    result = {
        "schema_version": 1,
        "definition": "provisional_surface_rhinoplasty_measurements_v1",
        "warning": "Definitions require surgeon review before clinical use.",
        "measurements": measurements,
    }
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    get_logger().info(
        "Measurement calculation complete | %s",
        " | ".join(f"{name}={value}" for name, value in measurements.items()),
    )
    return measurements
