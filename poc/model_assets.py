"""Download external inference assets used by the reconstruction pipeline."""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

from poc.logging_utils import get_logger

FACE_LANDMARKER_FILENAME = "face_landmarker.task"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MINIMUM_MODEL_BYTES = 1_000_000


def download_face_landmarker(output_dir: Path) -> Path:
    """Download the official MediaPipe face landmarker bundle atomically."""
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / FACE_LANDMARKER_FILENAME
    if destination.is_file() and destination.stat().st_size >= MINIMUM_MODEL_BYTES:
        get_logger().info("Face landmarker model already present | %s", destination)
        return destination

    temporary = destination.with_suffix(".download")
    temporary.unlink(missing_ok=True)
    get_logger().info("Downloading official MediaPipe face landmarker model")
    try:
        with (
            urllib.request.urlopen(FACE_LANDMARKER_URL, timeout=60) as response,
            temporary.open("wb") as file_handle,
        ):
            shutil.copyfileobj(response, file_handle)
        if temporary.stat().st_size < MINIMUM_MODEL_BYTES:
            raise RuntimeError(
                f"Downloaded face landmarker model is unexpectedly small: {temporary.stat().st_size} bytes"
            )
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    get_logger().info(
        "Face landmarker model ready | %s | %.1f MB",
        destination,
        destination.stat().st_size / 1_000_000,
    )
    return destination
