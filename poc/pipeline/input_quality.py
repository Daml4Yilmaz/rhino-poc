"""Retrospective image-quality metrics for standardized face captures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from poc.logging_utils import ProgressReporter


@dataclass(frozen=True)
class VideoQualitySeries:
    sharpness: np.ndarray
    median_luminance: np.ndarray
    dark_pixel_percent: np.ndarray
    bright_pixel_percent: np.ndarray


def decode_video_quality(video: Path, expected_frames: int) -> VideoQualitySeries:
    """Decode a video and score its central subject region consistently."""
    reader = cv2.VideoCapture(str(video))
    if not reader.isOpened():
        raise RuntimeError(f"Cannot open RGB video: {video}")
    sharpness: list[float] = []
    luminance: list[float] = []
    dark: list[float] = []
    bright: list[float] = []
    progress = ProgressReporter("Decode and score RGB frames", total=expected_frames)
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (0, 0), fx=0.35, fy=0.35, interpolation=cv2.INTER_AREA)
        height, width = gray.shape
        region = gray[
            round(height * 0.10) : round(height * 0.90),
            round(width * 0.15) : round(width * 0.85),
        ]
        sharpness.append(float(cv2.Laplacian(region, cv2.CV_64F).var()))
        luminance.append(float(np.median(region)))
        dark.append(float(np.mean(region <= 5) * 100.0))
        bright.append(float(np.mean(region >= 250) * 100.0))
        progress.update(len(sharpness))
    reader.release()
    progress.finish(detail=f"decoded {len(sharpness)} frames")
    if abs(len(sharpness) - expected_frames) > 1:
        raise RuntimeError(
            f"Decoded RGB frame count ({len(sharpness)}) does not match capture records "
            f"({expected_frames})"
        )
    return VideoQualitySeries(
        sharpness=np.asarray(sharpness, dtype=np.float64),
        median_luminance=np.asarray(luminance, dtype=np.float64),
        dark_pixel_percent=np.asarray(dark, dtype=np.float64),
        bright_pixel_percent=np.asarray(bright, dtype=np.float64),
    )


def summarize_video_quality(
    series: VideoQualitySeries, selected_indices: list[int] | None = None
) -> dict[str, float | int | str]:
    """Return robust metrics, optionally emphasizing reconstruction views."""
    if not len(series.sharpness):
        raise ValueError("No decoded frames are available for quality analysis")
    selected = np.asarray(
        selected_indices if selected_indices is not None else range(len(series.sharpness)),
        dtype=np.int64,
    )
    selected = selected[(selected >= 0) & (selected < len(series.sharpness))]
    if not len(selected):
        raise ValueError("No valid selected frame indices are available")
    luminance_p05, luminance_p95 = np.percentile(series.median_luminance, [5, 95])
    sharpness_p10, sharpness_median = np.percentile(series.sharpness[selected], [10, 50])
    return {
        "method": "central_roi_laplacian_and_luminance_v1",
        "decoded_frame_count": len(series.sharpness),
        "evaluated_frame_count": len(selected),
        "selected_sharpness_p10": round(float(sharpness_p10), 3),
        "selected_sharpness_median": round(float(sharpness_median), 3),
        "median_dark_pixel_percent": round(float(np.median(series.dark_pixel_percent)), 3),
        "median_bright_pixel_percent": round(float(np.median(series.bright_pixel_percent)), 3),
        "luminance_p05": round(float(luminance_p05), 3),
        "luminance_p95": round(float(luminance_p95), 3),
        "luminance_temporal_range": round(float(luminance_p95 - luminance_p05), 3),
    }
