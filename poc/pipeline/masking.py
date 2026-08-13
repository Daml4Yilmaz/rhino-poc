"""Create face masks and background-suppressed reconstruction images."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from poc.face_landmarker import FaceLandmarkDetector
from poc.logging_utils import ProgressReporter, get_logger


def _mask_from_landmarks(image: np.ndarray, landmarks) -> np.ndarray:
    height, width = image.shape[:2]
    points = np.asarray(
        [
            [
                np.clip(round(landmark.x * width), 0, width - 1),
                np.clip(round(landmark.y * height), 0, height - 1),
            ]
            for landmark in landmarks[:468]
        ],
        dtype=np.int32,
    )
    hull = cv2.convexHull(points)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    # Include the full skin boundary and a small safety margin for triangulation.
    diameter = max(9, round(max(width, height) * 0.025))
    if diameter % 2 == 0:
        diameter += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    return cv2.dilate(mask, kernel, iterations=1)


def run_masking(
    images_dir: Path,
    masks_dir: Path,
    reconstruction_images_dir: Path,
    model_path: Path,
    *,
    minimum_success_ratio: float = 0.70,
) -> dict[str, int | float]:
    """Mask selected frames and exclude views where no face can be localized."""
    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        raise RuntimeError(f"No selected images found in {images_dir}")
    masks_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_images_dir.mkdir(parents=True, exist_ok=True)

    progress = ProgressReporter("Face masking", total=len(image_paths))
    successes = 0
    failures: list[str] = []
    with FaceLandmarkDetector(model_path) as detector:
        for index, image_path in enumerate(image_paths, start=1):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                failures.append(image_path.name)
                progress.update(index)
                continue
            landmarks = detector.detect(image)
            if landmarks is None:
                failures.append(image_path.name)
                progress.update(index)
                continue

            mask = _mask_from_landmarks(image, landmarks)
            masked = cv2.bitwise_and(image, image, mask=mask)
            mask_path = masks_dir / f"{image_path.name}.png"
            if not cv2.imwrite(str(mask_path), mask):
                raise RuntimeError(f"Failed to write mask: {mask_path}")
            if not cv2.imwrite(
                str(reconstruction_images_dir / image_path.name),
                masked,
                [cv2.IMWRITE_JPEG_QUALITY, 96],
            ):
                raise RuntimeError(f"Failed to write masked image: {image_path.name}")
            successes += 1
            progress.update(index)
    progress.finish(detail=f"{successes} masks created")

    ratio = successes / len(image_paths)
    if ratio < minimum_success_ratio or successes < 60:
        raise RuntimeError(
            f"Face masking succeeded for only {successes}/{len(image_paths)} images "
            f"({ratio:.0%}). The capture cannot proceed reliably."
        )

    # Remove stale outputs for frames that failed in a previous run.
    for failed in failures:
        (masks_dir / f"{failed}.png").unlink(missing_ok=True)
        (reconstruction_images_dir / failed).unlink(missing_ok=True)

    get_logger().info(
        "Face masking complete | %d/%d accepted (%.0f%%) | %d rejected views",
        successes,
        len(image_paths),
        ratio * 100.0,
        len(failures),
    )
    return {
        "input_images": len(image_paths),
        "accepted_images": successes,
        "rejected_images": len(failures),
        "success_ratio": round(ratio, 4),
    }


def copy_without_masking(images_dir: Path, reconstruction_images_dir: Path) -> None:
    """Explicit diagnostic fallback; not suitable for a final face-only mesh."""
    reconstruction_images_dir.mkdir(parents=True, exist_ok=True)
    for image_path in images_dir.glob("*.jpg"):
        shutil.copy2(image_path, reconstruction_images_dir / image_path.name)
    get_logger().warning("Face masking was disabled; dense reconstruction will include the room")
