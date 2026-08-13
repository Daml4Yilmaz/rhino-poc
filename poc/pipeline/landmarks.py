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


def _bind_to_surface(
    points_m: dict[str, np.ndarray], mesh_path: Path
) -> tuple[dict[str, np.ndarray], dict[str, dict], dict[str, float]]:
    """Project points onto triangles and retain deformation-safe surface bindings."""
    import open3d as o3d

    legacy_mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(legacy_mesh.vertices, dtype=np.float64)
    triangles = np.asarray(legacy_mesh.triangles, dtype=np.int64)
    if not len(vertices) or not len(triangles):
        raise RuntimeError(f"Cannot bind landmarks to an empty mesh: {mesh_path}")

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy_mesh))
    names = list(points_m)
    queries = np.asarray([points_m[name] for name in names], dtype=np.float32)
    result = scene.compute_closest_points(o3d.core.Tensor(queries))
    closest = result["points"].numpy().astype(np.float64)
    triangle_ids = result["primitive_ids"].numpy().astype(np.int64)
    primitive_uvs = result["primitive_uvs"].numpy().astype(np.float64)

    snapped: dict[str, np.ndarray] = {}
    bindings: dict[str, dict] = {}
    distances_mm: dict[str, float] = {}
    for row, name in enumerate(names):
        triangle_id = int(triangle_ids[row])
        if triangle_id < 0 or triangle_id >= len(triangles):
            raise RuntimeError(f"No authoritative surface found for landmark '{name}'")
        u, v = primitive_uvs[row]
        barycentric = np.asarray([1.0 - u - v, u, v], dtype=np.float64)
        triangle = triangles[triangle_id]
        bound_point = barycentric @ vertices[triangle]
        if np.linalg.norm(bound_point - closest[row]) > 1e-6:
            raise RuntimeError(f"Invalid barycentric surface binding for landmark '{name}'")
        snapped[name] = bound_point
        bindings[name] = {
            "triangle_index": triangle_id,
            "barycentric": barycentric.round(10).tolist(),
            "position_m": bound_point.round(10).tolist(),
        }
        distances_mm[name] = float(np.linalg.norm(bound_point - points_m[name]) * 1000.0)
    return snapped, bindings, distances_mm


def triangulate_landmarks(
    images_dir: Path,
    sparse_model: Path,
    frame_index: Path,
    authoritative_mesh_path: Path,
    geometry_metadata_path: Path,
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

    geometry = json.loads(geometry_metadata_path.read_text(encoding="utf-8"))
    triangulated_m = {
        name: point * (scale_mm_per_unit / 1000.0) for name, point in triangulated.items()
    }
    snapped, surface_bindings, snap_distances_mm = _bind_to_surface(
        triangulated_m, authoritative_mesh_path
    )
    landmarks_mm = {name: (point * 1000.0).round(5).tolist() for name, point in snapped.items()}
    for name, distance_mm in snap_distances_mm.items():
        quality[name]["surface_snap_distance_mm"] = round(distance_mm, 3)
    result = {
        "schema_version": 2,
        "definition": "provisional_mediapipe_surface_landmarks_v2",
        "geometry_id": geometry["geometry_id"],
        "units": "millimetres",
        "landmarks": landmarks_mm,
        "surface_bindings": surface_bindings,
        "quality": quality,
    }
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    get_logger().info(
        "Landmark triangulation complete | %d landmarks | %d detected views",
        len(landmarks_mm),
        detected_images,
    )
    return result
