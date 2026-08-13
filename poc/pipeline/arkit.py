"""Read and validate metric iPhone capture data exported by Stray Scanner.

ARKit data supplies metric camera poses, camera intrinsics, and optional LiDAR
depth. It is never aligned to decoded RGB frames by an assumed frame rate. The
exported frame identifier is the authoritative synchronization key.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from poc.logging_utils import get_logger


@dataclass(frozen=True)
class CaptureFrame:
    frame_id: int
    timestamp: float
    center_m: np.ndarray
    quaternion_xyzw: np.ndarray
    intrinsics: np.ndarray


@dataclass
class ArkitCapture:
    """Normalized capture with one record for each exported RGB frame."""

    frames: list[CaptureFrame]
    rgb_path: Path
    rgb_size: tuple[int, int]
    depth_dir: Path | None
    confidence_dir: Path | None
    source: str = "stray"
    validation_summary: dict[str, float | int | bool] | None = None

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def frame_ids(self) -> np.ndarray:
        return np.asarray([frame.frame_id for frame in self.frames], dtype=np.int64)

    @property
    def timestamps(self) -> np.ndarray:
        return np.asarray([frame.timestamp for frame in self.frames], dtype=np.float64)

    @property
    def centers(self) -> np.ndarray:
        return np.asarray([frame.center_m for frame in self.frames], dtype=np.float64)

    @property
    def quaternions(self) -> np.ndarray:
        return np.asarray([frame.quaternion_xyzw for frame in self.frames], dtype=np.float64)

    @property
    def intrinsics(self) -> np.ndarray:
        return np.asarray([frame.intrinsics for frame in self.frames], dtype=np.float64)

    @property
    def has_depth(self) -> bool:
        return self.depth_dir is not None

    def frame(self, frame_id: int) -> CaptureFrame:
        if frame_id < 0 or frame_id >= self.n_frames:
            raise IndexError(f"Frame {frame_id} is outside the capture range")
        frame = self.frames[frame_id]
        if frame.frame_id != frame_id:
            raise KeyError(f"Capture has no contiguous record for frame {frame_id}")
        return frame

    def depth_m(self, frame_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        if self.depth_dir is None:
            return None
        depth_path = self.depth_dir / f"{frame_id:06d}.png"
        if not depth_path.exists():
            return None
        raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if raw_depth is None:
            return None
        depth = raw_depth.astype(np.float32) / 1000.0

        confidence = np.full(raw_depth.shape, 2, dtype=np.uint8)
        if self.confidence_dir is not None:
            confidence_path = self.confidence_dir / f"{frame_id:06d}.png"
            loaded = cv2.imread(str(confidence_path), cv2.IMREAD_UNCHANGED)
            if loaded is not None:
                confidence = loaded
        return depth, confidence


def _video_metadata(video: Path) -> tuple[tuple[int, int], int]:
    reader = cv2.VideoCapture(str(video))
    if not reader.isOpened():
        raise RuntimeError(f"Cannot open RGB video: {video}")
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    reader.release()
    return (width, height), frame_count


def load_capture(path: Path) -> ArkitCapture:
    """Load a Stray Scanner directory and enforce explicit stream alignment."""
    root = Path(path).expanduser().resolve()
    required = {
        "odometry": root / "odometry.csv",
        "intrinsics": root / "camera_matrix.csv",
        "RGB video": root / "rgb.mp4",
    }
    missing = [label for label, item in required.items() if not item.exists()]
    if missing:
        raise FileNotFoundError(f"Invalid Stray Scanner export at {root}; missing: {missing}")

    fallback_k = np.loadtxt(required["intrinsics"], delimiter=",").reshape(3, 3)
    frames: list[CaptureFrame] = []
    with required["odometry"].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        if not reader.fieldnames:
            raise RuntimeError(f"Odometry CSV has no header: {required['odometry']}")
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for raw in reader:
            row = {key.strip(): (value or "").strip() for key, value in raw.items()}
            frame_id = int(row["frame"])
            fx = float(row["fx"]) if row.get("fx") else float(fallback_k[0, 0])
            fy = float(row["fy"]) if row.get("fy") else float(fallback_k[1, 1])
            cx = float(row["cx"]) if row.get("cx") else float(fallback_k[0, 2])
            cy = float(row["cy"]) if row.get("cy") else float(fallback_k[1, 2])
            frames.append(
                CaptureFrame(
                    frame_id=frame_id,
                    timestamp=float(row["timestamp"]),
                    center_m=np.asarray([row["x"], row["y"], row["z"]], dtype=np.float64),
                    quaternion_xyzw=np.asarray(
                        [row["qx"], row["qy"], row["qz"], row["qw"]], dtype=np.float64
                    ),
                    intrinsics=np.asarray(
                        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                        dtype=np.float64,
                    ),
                )
            )

    if not frames:
        raise RuntimeError(f"Odometry CSV contains no frames: {required['odometry']}")
    expected_ids = list(range(len(frames)))
    actual_ids = [frame.frame_id for frame in frames]
    if actual_ids != expected_ids:
        raise RuntimeError(
            "Frame identifiers are not contiguous from zero. Explicit gap handling is required "
            "before this capture can be reconstructed."
        )

    rgb_size, video_frame_count = _video_metadata(required["RGB video"])
    if abs(video_frame_count - len(frames)) > 1:
        raise RuntimeError(
            f"RGB/odometry count mismatch: video={video_frame_count}, odometry={len(frames)}"
        )

    depth_dir = root / "depth" if (root / "depth").is_dir() else None
    confidence_dir = root / "confidence" if (root / "confidence").is_dir() else None
    capture = ArkitCapture(
        frames=frames,
        rgb_path=required["RGB video"],
        rgb_size=rgb_size,
        depth_dir=depth_dir,
        confidence_dir=confidence_dir,
    )
    capture.validation_summary = validate_capture(capture)
    return capture


def _view_directions(quaternions: np.ndarray) -> np.ndarray:
    """Return ARKit camera forward directions (-Z) in world coordinates."""
    x, y, z, w = [quaternions[:, index] for index in range(4)]
    norm_squared = x * x + y * y + z * z + w * w
    scale = 2.0 / np.where(norm_squared < 1e-12, 1.0, norm_squared)
    directions = np.stack(
        [
            -(scale * (x * z + y * w)),
            -(scale * (y * z - x * w)),
            -(1.0 - scale * (x * x + y * y)),
        ],
        axis=1,
    )
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def orientation_coverage_degrees(capture: ArkitCapture) -> float:
    directions = _view_directions(capture.quaternions)
    similarities = np.clip(directions @ directions.T, -1.0, 1.0)
    return float(np.degrees(np.arccos(similarities)).max())


def validate_capture(capture: ArkitCapture) -> dict[str, float | int | bool]:
    """Validate synchronization, trajectory continuity, and useful translation."""
    if capture.n_frames < 120:
        raise RuntimeError(f"Capture has only {capture.n_frames} frames; at least 120 are required")

    timestamps = capture.timestamps
    intervals = np.diff(timestamps)
    if np.any(intervals <= 0):
        raise RuntimeError("Capture timestamps are not strictly increasing")

    centers = capture.centers
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    jump_count = int(np.count_nonzero(steps > 0.15))
    if jump_count > max(1, round(capture.n_frames * 0.02)):
        raise RuntimeError(
            f"ARKit tracking is discontinuous: {jump_count} frame-to-frame jumps exceed 15 cm"
        )

    path_length = float(steps.sum())
    trajectory_span = float(np.linalg.norm(np.ptp(centers, axis=0)))
    coverage = orientation_coverage_degrees(capture)
    duration = float(timestamps[-1] - timestamps[0])
    effective_fps = float((capture.n_frames - 1) / duration)
    focal_lengths = capture.intrinsics[:, 0, 0]
    focal_drift = float(np.ptp(focal_lengths) / np.mean(focal_lengths) * 100.0)

    # Orientation alone cannot prove parallax: a stationary camera can pan.
    if trajectory_span < 0.25 or path_length < 0.75:
        raise RuntimeError(
            f"Camera translation is insufficient for reconstruction: path={path_length:.2f} m, "
            f"span={trajectory_span:.2f} m"
        )

    summary: dict[str, float | int | bool] = {
        "frame_count": capture.n_frames,
        "duration_seconds": round(duration, 3),
        "effective_fps": round(effective_fps, 3),
        "path_length_m": round(path_length, 3),
        "trajectory_span_m": round(trajectory_span, 3),
        "orientation_coverage_degrees": round(coverage, 2),
        "tracking_jump_count": jump_count,
        "focal_length_drift_percent": round(focal_drift, 3),
        "has_depth": capture.has_depth,
    }
    get_logger().info(
        "Capture validated | %d frames | %.1fs at %.1f effective fps | %.2fm path | "
        "%.1f° coverage | LiDAR %s",
        capture.n_frames,
        duration,
        effective_fps,
        path_length,
        coverage,
        "available" if capture.has_depth else "unavailable",
    )
    return summary
