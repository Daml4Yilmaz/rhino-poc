"""Triangulate provisional anatomical landmarks from registered RGB views."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from poc.face_landmarker import FaceLandmarkDetector
from poc.logging_utils import ProgressReporter, get_logger

# Provisional MediaPipe-to-anatomy mapping for the surface-only PoC. These
# definitions are intentionally centralized so surgeon review can replace them.
LANDMARK_INDICES = {
    "glabella": 9,
    "nasion": 168,
    "pronasale": 1,
    "subnasale": 2,
    "columella": 164,
    "labiale_superius": 0,
    "left_alare": 98,
    "right_alare": 327,
    "left_endocanthion": 133,
    "right_endocanthion": 362,
}


def _camera_from_world(image):
    transform = image.cam_from_world
    return transform() if callable(transform) else transform


def _closest_point_to_rays(centers: np.ndarray, directions: np.ndarray) -> np.ndarray:
    identity = np.eye(3)
    matrices = identity[None, :, :] - directions[:, :, None] * directions[:, None, :]
    left = matrices.sum(axis=0)
    right = np.einsum("nij,nj->i", matrices, centers)
    return np.linalg.solve(left, right)


def _robust_triangulation(
    centers: np.ndarray, directions: np.ndarray, minimum_views: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    active = np.ones(len(centers), dtype=bool)
    for _ in range(4):
        if active.sum() < minimum_views:
            break
        point = _closest_point_to_rays(centers[active], directions[active])
        offsets = point - centers
        distances = np.linalg.norm(np.cross(offsets, directions), axis=1)
        median = float(np.median(distances[active]))
        mad = float(np.median(np.abs(distances[active] - median)))
        threshold = max(median + 3.0 * mad, median * 2.0, 1e-6)
        updated = distances <= threshold
        if np.array_equal(updated, active):
            break
        active = updated
    if active.sum() < minimum_views:
        raise RuntimeError(f"Landmark triangulation retained only {active.sum()} views")
    return _closest_point_to_rays(centers[active], directions[active]), active


def _snap_to_mesh(points: dict[str, np.ndarray], mesh_path: Path) -> dict[str, np.ndarray]:
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices)
    if not len(vertices):
        raise RuntimeError(f"Cannot snap landmarks to an empty mesh: {mesh_path}")
    tree = o3d.geometry.KDTreeFlann(mesh)
    snapped: dict[str, np.ndarray] = {}
    for name, point in points.items():
        _, indices, _ = tree.search_knn_vector_3d(point, 1)
        snapped[name] = vertices[indices[0]] if indices else point
    return snapped


def triangulate_landmarks(
    images_dir: Path,
    sparse_model: Path,
    frame_index: Path,
    mesh_path: Path,
    scale_mm_per_unit: float,
    output_json: Path,
    model_path: Path,
) -> dict:
    """Detect landmarks in registered views and triangulate them in 3D."""
    import pycolmap

    metadata = json.loads(frame_index.read_text(encoding="utf-8"))["images"]
    reconstruction = pycolmap.Reconstruction(str(sparse_model))
    registered = {image.name: image for image in reconstruction.images.values()}
    observations: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        name: [] for name in LANDMARK_INDICES
    }
    image_paths = [images_dir / name for name in sorted(registered) if (images_dir / name).exists()]
    progress = ProgressReporter("Multi-view landmark detection", total=len(image_paths))
    detected_images = 0
    with FaceLandmarkDetector(model_path) as detector:
        for index, image_path in enumerate(image_paths, start=1):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            face = detector.detect(image)
            if face is None:
                progress.update(index)
                continue
            detected_images += 1
            model_image = registered[image_path.name]
            camera_from_world = _camera_from_world(model_image)
            center = np.asarray(camera_from_world.inverse().translation, dtype=np.float64)
            rotation = np.asarray(camera_from_world.rotation.matrix(), dtype=np.float64)
            intrinsics = np.asarray(metadata[image_path.name]["intrinsics"], dtype=np.float64)
            inverse_intrinsics = np.linalg.inv(intrinsics)
            height, width = image.shape[:2]
            for name, landmark_index in LANDMARK_INDICES.items():
                landmark = face[landmark_index]
                pixel = np.asarray([landmark.x * width, landmark.y * height, 1.0])
                camera_ray = inverse_intrinsics @ pixel
                world_ray = rotation.T @ camera_ray
                world_ray /= np.linalg.norm(world_ray)
                observations[name].append((center, world_ray))
            progress.update(index)
    progress.finish(detail=f"face detected in {detected_images} registered views")

    triangulated: dict[str, np.ndarray] = {}
    quality: dict[str, dict] = {}
    for name, rays in observations.items():
        if len(rays) < 6:
            raise RuntimeError(f"Landmark '{name}' was observed in only {len(rays)} views")
        centers = np.asarray([item[0] for item in rays])
        directions = np.asarray([item[1] for item in rays])
        point, inliers = _robust_triangulation(centers, directions)
        triangulated[name] = point
        residuals = np.linalg.norm(np.cross(point - centers, directions), axis=1)
        quality[name] = {
            "observations": len(rays),
            "inliers": int(inliers.sum()),
            "median_ray_residual_mm": round(
                float(np.median(residuals[inliers])) * scale_mm_per_unit, 3
            ),
        }

    snapped = _snap_to_mesh(triangulated, mesh_path)
    landmarks_mm = {
        name: (point * scale_mm_per_unit).round(5).tolist() for name, point in snapped.items()
    }
    result = {
        "schema_version": 1,
        "definition": "provisional_mediapipe_surface_landmarks_v1",
        "units": "millimetres",
        "landmarks": landmarks_mm,
        "quality": quality,
    }
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    get_logger().info(
        "Landmark triangulation complete | %d landmarks | %d detected views",
        len(landmarks_mm),
        detected_images,
    )
    return result
