"""Time-bounded dense facial reconstruction with COLMAP PatchMatch."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from poc.logging_utils import format_duration, get_logger, run_command


def _automatic_cache_gb() -> int:
    try:
        total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return max(2, min(8, int(total_bytes / 1e9 * 0.30)))
    except (ValueError, OSError, AttributeError):
        return 4


def _require_cuda_colmap(colmap_binary: str) -> None:
    result = subprocess.run(
        [colmap_binary, "-h"], capture_output=True, text=True, timeout=60, check=False
    )
    output = result.stdout + result.stderr
    if "without CUDA" in output:
        raise RuntimeError(
            "Dense PatchMatch requires a CUDA-enabled COLMAP build. Run this stage in the "
            "provided Colab notebook with a T4 GPU."
        )


def _configure_views(config_path: Path, source_images: int, maximum_references: int) -> int:
    if not config_path.exists():
        raise FileNotFoundError(f"PatchMatch configuration was not generated: {config_path}")
    lines = [line for line in config_path.read_text(encoding="utf-8").splitlines() if line]
    blocks = [(lines[index], lines[index + 1]) for index in range(0, len(lines) - 1, 2)]
    original_count = len(blocks)
    if not blocks:
        raise RuntimeError(f"PatchMatch configuration is empty: {config_path}")

    if maximum_references and original_count > maximum_references:
        keep = np.linspace(0, original_count - 1, maximum_references).round().astype(int)
        blocks = [blocks[index] for index in sorted(set(keep.tolist()))]
    configured: list[str] = []
    for image_name, neighbors in blocks:
        configured.append(image_name)
        configured.append(re.sub(r"__auto__,\s*\d+", f"__auto__, {source_images}", neighbors))
    config_path.write_text("\n".join(configured) + "\n", encoding="utf-8")

    fusion_config = config_path.parent / "fusion.cfg"
    if fusion_config.exists():
        fusion_config.write_text(
            "\n".join(image_name for image_name, _ in blocks) + "\n", encoding="utf-8"
        )
    get_logger().info(
        "MVS view budget | %d/%d reference images | %d source images per reference",
        len(blocks),
        original_count,
        source_images,
    )
    return len(blocks)


def _depth_map_probe(depth_dir: Path, suffix: str, total: int):
    def probe() -> tuple[int, int, str]:
        count = len(list(depth_dir.glob(f"*{suffix}"))) if depth_dir.is_dir() else 0
        return count, total, "depth maps written"

    return probe


def _keep_largest_component(mesh):

    triangle_clusters, cluster_counts, _ = mesh.cluster_connected_triangles()
    cluster_counts = np.asarray(cluster_counts)
    if not len(cluster_counts):
        return mesh
    labels = np.asarray(triangle_clusters)
    largest = int(np.argmax(cluster_counts))
    mesh.remove_triangles_by_mask(labels != largest)
    mesh.remove_unreferenced_vertices()
    return mesh


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _point_cloud_diagnostics(
    point_cloud,
    path: Path,
    *,
    scale_mm_per_unit: float,
) -> dict:
    """Describe the unmodified stereo-fusion artifact before Poisson processing."""
    points = np.asarray(point_cloud.points, dtype=np.float64)
    if not len(points):
        raise RuntimeError(f"Stereo fusion produced an empty point cloud: {path}")
    finite_rows = np.all(np.isfinite(points), axis=1)
    finite_points = points[finite_rows]
    if not len(finite_points):
        raise RuntimeError(f"Stereo fusion produced no finite 3D points: {path}")
    minimum = finite_points.min(axis=0)
    maximum = finite_points.max(axis=0)
    extent = maximum - minimum
    return {
        "schema_version": 1,
        "role": "raw_fused_dense_point_cloud_pre_poisson",
        "path": path.name,
        "absolute_path_at_generation": str(path.resolve()),
        "generated_by": "COLMAP stereo_fusion",
        "source_stage": "mvs_stereo_fusion",
        "geometry_state": "direct_stereo_fusion_output_before_normal_estimation_and_poisson",
        "coordinate_system": "COLMAP world coordinates",
        "coordinate_units": "COLMAP reconstruction units",
        "point_count": len(points),
        "nonfinite_point_count": int(len(points) - np.count_nonzero(finite_rows)),
        "normals_present_in_persisted_artifact": bool(point_cloud.has_normals()),
        "colors_present_in_persisted_artifact": bool(point_cloud.has_colors()),
        "bounding_box_min": minimum.round(9).tolist(),
        "bounding_box_max": maximum.round(9).tolist(),
        "bounding_box_extent": extent.round(9).tolist(),
        "metric_bounding_box_extent_mm": (extent * scale_mm_per_unit).round(4).tolist(),
        "scale_mm_per_unit_for_diagnostics": float(scale_mm_per_unit),
        "file_size_bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def run_mvs(
    images_dir: Path,
    sparse_model: Path,
    dense_dir: Path,
    output_mesh: Path,
    fused_point_cloud_path: Path,
    metrics_path: Path,
    *,
    scale_mm_per_unit: float,
    colmap_binary: str = "colmap",
    cache_size_gb: int | None = None,
    max_image_size: int | None = None,
    source_images: int = 6,
    maximum_references: int = 96,
    geometric_consistency: bool = False,
    poisson_depth: int = 9,
    poisson_trim_percent: float = 4.0,
) -> dict:
    """Run dense reconstruction with defaults designed for a sub-hour T4 budget."""
    _require_cuda_colmap(colmap_binary)
    if dense_dir.exists():
        shutil.rmtree(dense_dir)
    dense_dir.mkdir(parents=True)
    raw_log = dense_dir.parent / "colmap.log"

    run_command(
        [
            colmap_binary,
            "image_undistorter",
            "--image_path",
            str(images_dir),
            "--input_path",
            str(sparse_model),
            "--output_path",
            str(dense_dir),
            "--output_type",
            "COLMAP",
        ],
        stage="MVS image preparation",
        raw_log_file=raw_log,
    )

    reference_count = _configure_views(
        dense_dir / "stereo" / "patch-match.cfg", source_images, maximum_references
    )
    cache_gb = cache_size_gb or _automatic_cache_gb()
    estimated_minutes = reference_count * 10.0 / 60.0
    if geometric_consistency:
        estimated_minutes *= 3.5
    get_logger().info(
        "MVS estimate | approximately %.0f minutes on a T4; this is an estimate, not a gate",
        estimated_minutes,
    )

    patchmatch_command = [
        colmap_binary,
        "patch_match_stereo",
        "--workspace_path",
        str(dense_dir),
        "--PatchMatchStereo.geom_consistency",
        "1" if geometric_consistency else "0",
        "--PatchMatchStereo.cache_size",
        str(cache_gb),
    ]
    if max_image_size:
        patchmatch_command.extend(["--PatchMatchStereo.max_image_size", str(max_image_size)])
    suffix = ".geometric.bin" if geometric_consistency else ".photometric.bin"
    run_command(
        patchmatch_command,
        stage="MVS PatchMatch",
        raw_log_file=raw_log,
        progress_probe=_depth_map_probe(
            dense_dir / "stereo" / "depth_maps", suffix, reference_count
        ),
    )

    # This path is intentionally outside dense_dir. dense_dir is a disposable
    # COLMAP workspace, while this exact stereo_fusion output is a permanent
    # diagnostic artifact and must survive workspace cleanup.
    fused_point_cloud_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            colmap_binary,
            "stereo_fusion",
            "--workspace_path",
            str(dense_dir),
            "--output_path",
            str(fused_point_cloud_path),
            "--input_type",
            "geometric" if geometric_consistency else "photometric",
            "--StereoFusion.use_cache",
            "1",
            "--StereoFusion.cache_size",
            str(cache_gb),
        ],
        stage="MVS depth fusion",
        raw_log_file=raw_log,
    )

    import open3d as o3d

    started_at = time.monotonic()
    if not fused_point_cloud_path.is_file():
        raise RuntimeError(
            "COLMAP stereo fusion completed without its declared point-cloud output: "
            f"{fused_point_cloud_path}"
        )
    point_cloud = o3d.io.read_point_cloud(str(fused_point_cloud_path))
    diagnostics = _point_cloud_diagnostics(
        point_cloud,
        fused_point_cloud_path,
        scale_mm_per_unit=scale_mm_per_unit,
    )
    diagnostics["normals_estimated_in_memory_for_poisson"] = False
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    if diagnostics["nonfinite_point_count"]:
        raise RuntimeError(
            "Dense fusion produced non-finite points; inspect the preserved point cloud and "
            f"{metrics_path} before retrying"
        )
    if len(point_cloud.points) < 10_000:
        raise RuntimeError(
            f"Dense fusion produced only {len(point_cloud.points)} points; no usable mesh can be built"
        )
    get_logger().info(
        "Stereo fusion preserved | %,d points | %.1f MB | normals %s | %s",
        len(point_cloud.points),
        diagnostics["file_size_bytes"] / 1_000_000.0,
        "present" if diagnostics["normals_present_in_persisted_artifact"] else "absent",
        fused_point_cloud_path,
    )
    normals_estimated_for_poisson = not point_cloud.has_normals()
    if normals_estimated_for_poisson:
        point_cloud.estimate_normals()
    diagnostics["normals_estimated_in_memory_for_poisson"] = normals_estimated_for_poisson
    metrics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    get_logger().info(
        "Surface reconstruction | %,d fused points | Poisson depth %d",
        len(point_cloud.points),
        poisson_depth,
    )
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud, depth=poisson_depth
    )
    density_values = np.asarray(densities)
    cutoff = np.percentile(density_values, poisson_trim_percent)
    mesh.remove_vertices_by_mask(density_values < cutoff)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh = _keep_largest_component(mesh)
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output_mesh), mesh):
        raise RuntimeError(f"Failed to write reconstructed mesh: {output_mesh}")
    get_logger().info(
        "Surface reconstruction complete | %,d vertices | %,d triangles | %s | %s",
        len(mesh.vertices),
        len(mesh.triangles),
        "vertex color available" if mesh.has_vertex_colors() else "no vertex color",
        format_duration(time.monotonic() - started_at),
    )
    return diagnostics
