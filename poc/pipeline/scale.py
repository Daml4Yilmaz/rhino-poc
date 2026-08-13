"""Estimate metric scale from ARKit trajectory and LiDAR depth."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from poc.logging_utils import get_logger

from .arkit import ArkitCapture

RANDOM_GENERATOR = np.random.default_rng(20260813)


def _camera_from_world(image):
    transform = image.cam_from_world
    return transform() if callable(transform) else transform


def umeyama_similarity(
    source: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Solve ``target ~= scale * rotation @ source + translation``."""
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    centered_source = source - source_mean
    centered_target = target - target_mean
    covariance = centered_target.T @ centered_source / len(source)
    left, singular_values, right_transposed = np.linalg.svd(covariance)
    reflection = np.eye(3)
    if np.linalg.det(left) * np.linalg.det(right_transposed) < 0:
        reflection[2, 2] = -1.0
    rotation = left @ reflection @ right_transposed
    variance = np.sum(centered_source**2) / len(source)
    scale = float(np.trace(np.diag(singular_values) @ reflection) / variance)
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def _ransac_similarity(
    source: np.ndarray,
    target: np.ndarray,
    *,
    iterations: int = 1000,
    threshold_m: float = 0.025,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    if len(source) < 6:
        raise ValueError("At least six trajectory correspondences are required")
    best_inliers = np.zeros(len(source), dtype=bool)
    best_median = float("inf")
    sample_size = min(8, len(source))
    for _ in range(iterations):
        sample = RANDOM_GENERATOR.choice(len(source), size=sample_size, replace=False)
        try:
            scale, rotation, translation = umeyama_similarity(source[sample], target[sample])
        except np.linalg.LinAlgError:
            continue
        predicted = scale * (source @ rotation.T) + translation
        residuals = np.linalg.norm(target - predicted, axis=1)
        inliers = residuals <= threshold_m
        median = float(np.median(residuals[inliers])) if np.any(inliers) else float("inf")
        if inliers.sum() > best_inliers.sum() or (
            inliers.sum() == best_inliers.sum() and median < best_median
        ):
            best_inliers = inliers
            best_median = median
    if best_inliers.sum() < max(20, round(len(source) * 0.50)):
        raise RuntimeError(
            f"Trajectory scale alignment found only {best_inliers.sum()}/{len(source)} inliers"
        )
    scale, rotation, translation = umeyama_similarity(source[best_inliers], target[best_inliers])
    return scale, rotation, translation, best_inliers


def scale_from_poses(reconstruction, capture: ArkitCapture, image_metadata: dict) -> dict:
    colmap_centers: list[np.ndarray] = []
    arkit_centers: list[np.ndarray] = []
    for image in reconstruction.images.values():
        entry = image_metadata.get(image.name)
        if entry is None:
            continue
        frame_id = int(entry["frame_id"])
        colmap_centers.append(_camera_from_world(image).inverse().translation)
        arkit_centers.append(capture.frame(frame_id).center_m)
    if len(colmap_centers) < 30:
        raise RuntimeError(f"Only {len(colmap_centers)} images have both COLMAP and ARKit poses")
    source = np.asarray(colmap_centers, dtype=np.float64)
    target = np.asarray(arkit_centers, dtype=np.float64)
    scale, rotation, translation, inliers = _ransac_similarity(source, target)
    residuals = np.linalg.norm(target - (scale * (source @ rotation.T) + translation), axis=1)
    return {
        "pose_scale_m_per_unit": scale,
        "pose_inlier_count": int(inliers.sum()),
        "pose_pair_count": len(source),
        "pose_inlier_ratio": round(float(inliers.mean()), 4),
        "pose_residual_median_mm": round(float(np.median(residuals[inliers])) * 1000.0, 3),
        "pose_residual_p95_mm": round(float(np.percentile(residuals[inliers], 95)) * 1000.0, 3),
    }


def scale_from_depth(
    reconstruction,
    capture: ArkitCapture,
    image_metadata: dict,
    masks_dir: Path | None,
    *,
    minimum_confidence: int = 2,
) -> dict | None:
    if not capture.has_depth:
        return None
    source_width, source_height = capture.rgb_size
    ratios: list[np.ndarray] = []
    used_images = 0
    for image in reconstruction.images.values():
        entry = image_metadata.get(image.name)
        if entry is None:
            continue
        frame_id = int(entry["frame_id"])
        depth_record = capture.depth_m(frame_id)
        if depth_record is None:
            continue
        depth, confidence = depth_record
        depth_height, depth_width = depth.shape
        image_to_source = np.linalg.inv(np.asarray(entry["source_to_image"], dtype=np.float64))
        mask = None
        if masks_dir is not None:
            mask = cv2.imread(str(masks_dir / f"{image.name}.png"), cv2.IMREAD_GRAYSCALE)

        camera_from_world = _camera_from_world(image)
        image_ratios: list[float] = []
        for point in image.points2D:
            if not point.has_point3D():
                continue
            world_point = reconstruction.points3D[point.point3D_id].xyz
            colmap_depth = float(
                (camera_from_world.rotation.matrix() @ world_point + camera_from_world.translation)[
                    2
                ]
            )
            if colmap_depth <= 0:
                continue
            image_u, image_v = point.xy
            if mask is not None:
                mask_x = int(np.clip(round(image_u), 0, mask.shape[1] - 1))
                mask_y = int(np.clip(round(image_v), 0, mask.shape[0] - 1))
                if mask[mask_y, mask_x] < 128:
                    continue
            source_pixel = image_to_source @ np.asarray([image_u, image_v, 1.0])
            source_u, source_v = source_pixel[:2] / source_pixel[2]
            if not (0 <= source_u < source_width and 0 <= source_v < source_height):
                continue
            depth_x = int(np.clip(source_u / source_width * depth_width, 0, depth_width - 1))
            depth_y = int(np.clip(source_v / source_height * depth_height, 0, depth_height - 1))
            lidar_depth = float(depth[depth_y, depth_x])
            if confidence[depth_y, depth_x] < minimum_confidence:
                continue
            if not 0.25 <= lidar_depth <= 1.5:
                continue
            image_ratios.append(lidar_depth / colmap_depth)
        if len(image_ratios) >= 15:
            ratios.append(np.asarray(image_ratios, dtype=np.float64))
            used_images += 1

    if used_images < 15:
        get_logger().warning(
            "LiDAR scale check unavailable | only %d images had sufficient face correspondences",
            used_images,
        )
        return None
    values = np.concatenate(ratios)
    median = float(np.median(values))
    absolute_deviation = np.abs(values - median)
    mad = float(np.median(absolute_deviation))
    robust_values = values[absolute_deviation <= max(3.5 * mad, median * 0.01)]
    scale = float(np.median(robust_values))
    return {
        "depth_scale_m_per_unit": scale,
        "depth_sample_count": len(robust_values),
        "depth_image_count": used_images,
        "depth_mad_percent": round(
            float(np.median(np.abs(robust_values - scale))) / scale * 100, 3
        ),
    }


def compute_scale(
    sparse_model: Path,
    capture: ArkitCapture,
    frame_index: Path,
    output_json: Path,
    *,
    masks_dir: Path | None = None,
    maximum_disagreement_percent: float = 2.0,
) -> float:
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(sparse_model))
    image_metadata = json.loads(frame_index.read_text(encoding="utf-8"))["images"]
    pose_result = scale_from_poses(reconstruction, capture, image_metadata)
    depth_result = scale_from_depth(reconstruction, capture, image_metadata, masks_dir=masks_dir)
    pose_scale = float(pose_result["pose_scale_m_per_unit"])
    result: dict = dict(pose_result)
    result["scale_source"] = "arkit_trajectory"
    if depth_result is not None:
        result.update(depth_result)
        disagreement = (
            abs(float(depth_result["depth_scale_m_per_unit"]) - pose_scale) / pose_scale * 100
        )
        result["scale_disagreement_percent"] = round(disagreement, 3)
        result["scale_verified"] = disagreement <= maximum_disagreement_percent
    else:
        result["scale_disagreement_percent"] = None
        result["scale_verified"] = None
    scale_mm = pose_scale * 1000.0
    result["scale_mm_per_unit"] = scale_mm
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    get_logger().info(
        "Metric scale | %.6f mm/unit | pose inliers %d/%d | LiDAR agreement %s",
        scale_mm,
        pose_result["pose_inlier_count"],
        pose_result["pose_pair_count"],
        (
            f"{result['scale_disagreement_percent']:.2f}%"
            if result["scale_disagreement_percent"] is not None
            else "unavailable"
        ),
    )
    if result["scale_verified"] is False:
        get_logger().warning(
            "Scale verification failed: pose and LiDAR estimates differ by %.2f%%",
            result["scale_disagreement_percent"],
        )
    return scale_mm
