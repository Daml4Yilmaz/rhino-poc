"""Headless, CPU-only MediaPipe face landmark detection."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import cv2
import numpy as np


class FaceLandmarkDetector:
    """Small context-managed wrapper around the MediaPipe Tasks API.

    The legacy MediaPipe Solutions API may attempt to initialize an OpenGL context even when
    processing still images. Explicitly selecting the CPU Tasks delegate keeps notebook and
    headless execution deterministic.
    """

    def __init__(self, model_path: Path) -> None:
        model_path = model_path.expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Face landmarker model not found: {model_path}. "
                "Run 'poc download-models' or follow the notebook setup cell."
            )
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - environment-specific dependency error
            raise RuntimeError("MediaPipe is required for face landmark detection") from exc

        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        )
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.45,
            min_face_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        self._mp = mp
        self._detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect(self, image_bgr: np.ndarray) -> list | None:
        """Return the first face's normalized landmarks, or ``None`` when no face is found."""
        if image_bgr is None or image_bgr.size == 0:
            return None
        image_rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        media_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=image_rgb,
        )
        result = self._detector.detect(media_image)
        return result.face_landmarks[0] if result.face_landmarks else None

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
