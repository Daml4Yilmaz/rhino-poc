"""Select sharp, temporally distributed frames and normalize pixel geometry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from poc.logging_utils import ProgressReporter, get_logger

from .arkit import ArkitCapture

ROTATIONS = {"none", "clockwise", "counterclockwise"}


@dataclass(frozen=True)
class ImageTransform:
    source_to_image: np.ndarray
    output_size: tuple[int, int]


def rotation_transform(source_size: tuple[int, int], rotation: str) -> ImageTransform:
    """Return the homogeneous source-pixel to oriented-image transform."""
    width, height = source_size
    if rotation == "none":
        matrix = np.eye(3, dtype=np.float64)
        output_size = (width, height)
    elif rotation == "clockwise":
        # (u, v) -> (height - 1 - v, u)
        matrix = np.asarray([[0.0, -1.0, height - 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        output_size = (height, width)
    elif rotation == "counterclockwise":
        # (u, v) -> (v, width - 1 - u)
        matrix = np.asarray([[0.0, 1.0, 0.0], [-1.0, 0.0, width - 1.0], [0.0, 0.0, 1.0]])
        output_size = (height, width)
    else:
        raise ValueError(f"Unsupported rotation '{rotation}'; expected one of {sorted(ROTATIONS)}")
    return ImageTransform(matrix, output_size)


def _apply_rotation(image: np.ndarray, rotation: str) -> np.ndarray:
    if rotation == "clockwise":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == "counterclockwise":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def transform_intrinsics(
    intrinsics: np.ndarray,
    source_size: tuple[int, int],
    rotation: str,
    scale: float,
) -> np.ndarray:
    """Return a conventional pinhole matrix for the oriented image.

    A raw homogeneous pixel rotation contains off-diagonal focal terms. COLMAP's
    PINHOLE model instead represents the equivalent rolled camera coordinate
    system with positive ``fx`` and ``fy`` on the diagonal.
    """
    width, height = source_size
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    if rotation == "none":
        oriented = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    elif rotation == "clockwise":
        oriented = np.asarray([[fy, 0.0, height - 1.0 - cy], [0.0, fx, cx], [0.0, 0.0, 1.0]])
    elif rotation == "counterclockwise":
        oriented = np.asarray([[fy, 0.0, cy], [0.0, fx, width - 1.0 - cx], [0.0, 0.0, 1.0]])
    else:
        raise ValueError(f"Unsupported rotation '{rotation}'")
    oriented[:2, :] *= scale
    oriented[2, 2] = 1.0
    return oriented


def _decode_sharpness(video: Path, expected_frames: int) -> np.ndarray:
    reader = cv2.VideoCapture(str(video))
    if not reader.isOpened():
        raise RuntimeError(f"Cannot open RGB video: {video}")
    scores: list[float] = []
    progress = ProgressReporter("Decode and score RGB frames", total=expected_frames)
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (0, 0), fx=0.35, fy=0.35, interpolation=cv2.INTER_AREA)
        scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        progress.update(len(scores))
    reader.release()
    progress.finish(detail=f"decoded {len(scores)} frames")
    if abs(len(scores) - expected_frames) > 1:
        raise RuntimeError(
            f"Decoded RGB frame count ({len(scores)}) does not match capture records "
            f"({expected_frames})"
        )
    return np.asarray(scores, dtype=np.float64)


def _select_indices(
    scores: np.ndarray,
    timestamps: np.ndarray,
    target_count: int,
    minimum_sharpness: float,
) -> list[int]:
    """Pick the sharpest frame in equal-duration windows."""
    count = min(target_count, len(scores))
    edges = np.linspace(timestamps[0], timestamps[-1] + 1e-9, count + 1)
    selected: list[int] = []
    for start, end in pairwise(edges):
        candidates = np.flatnonzero((timestamps >= start) & (timestamps < end))
        if not len(candidates):
            continue
        best = int(candidates[np.argmax(scores[candidates])])
        if scores[best] >= minimum_sharpness:
            selected.append(best)
    if len(selected) < 60:
        raise RuntimeError(
            f"Only {len(selected)} sharp frames were found. Improve lighting and move the phone "
            "more slowly before repeating the capture."
        )
    return selected


def select_frames(
    capture: ArkitCapture,
    output_dir: Path,
    index_path: Path,
    *,
    target_count: int = 120,
    minimum_sharpness: float = 35.0,
    max_dimension: int | None = 1400,
    rotation: str = "clockwise",
) -> list[str]:
    """Decode selected frames and write an explicit pixel transform for each image."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = _decode_sharpness(capture.rgb_path, capture.n_frames)
    usable_count = min(len(scores), capture.n_frames)
    selected = _select_indices(
        scores[:usable_count], capture.timestamps[:usable_count], target_count, minimum_sharpness
    )

    base_transform = rotation_transform(capture.rgb_size, rotation)
    oriented_width, oriented_height = base_transform.output_size
    scale = 1.0
    if max_dimension and max(oriented_width, oriented_height) > max_dimension:
        scale = max_dimension / float(max(oriented_width, oriented_height))
    output_size = (round(oriented_width * scale), round(oriented_height * scale))
    resize_transform = np.asarray([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]])
    source_to_image = resize_transform @ base_transform.source_to_image

    reader = cv2.VideoCapture(str(capture.rgb_path))
    wanted = set(selected)
    entries: dict[str, dict] = {}
    progress = ProgressReporter("Write selected RGB frames", total=len(selected))
    frame_index = 0
    written = 0
    while True:
        ok, image = reader.read()
        if not ok:
            break
        if frame_index in wanted:
            image = _apply_rotation(image, rotation)
            if scale != 1.0:
                image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
            name = f"frame_{frame_index:06d}.jpg"
            destination = output_dir / name
            if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 96]):
                raise RuntimeError(f"Failed to write selected frame: {destination}")

            effective_k = transform_intrinsics(
                capture.frame(frame_index).intrinsics,
                capture.rgb_size,
                rotation,
                scale,
            )
            entries[name] = {
                "frame_id": frame_index,
                "timestamp": capture.frame(frame_index).timestamp,
                "sharpness": round(float(scores[frame_index]), 3),
                "source_to_image": source_to_image.tolist(),
                "intrinsics": effective_k.tolist(),
                "image_size": list(output_size),
            }
            written += 1
            progress.update(written)
        frame_index += 1
    reader.release()
    progress.finish()

    if written != len(selected):
        raise RuntimeError(f"Expected to write {len(selected)} frames but wrote {written}")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "source_video": str(capture.rgb_path),
                "source_size": list(capture.rgb_size),
                "rotation": rotation,
                "target_count": target_count,
                "selected_count": written,
                "images": entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    get_logger().info(
        "Frame selection complete | %d/%d selected | output %dx%d | median sharpness %.1f",
        written,
        capture.n_frames,
        output_size[0],
        output_size[1],
        float(np.median(scores[selected])),
    )
    return list(entries)
