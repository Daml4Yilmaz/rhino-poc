"""Sparse visual reconstruction with measured per-frame camera intrinsics."""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path

import numpy as np

from poc.logging_utils import get_logger, run_command

PINHOLE_MODEL_ID = 1
COLMAP_PROGRESS = re.compile(
    r"(?:Processing|Registering|Matching).*?(?P<current>\d+)\s*/\s*(?P<total>\d+)", re.IGNORECASE
)


def _gpu_option_names(colmap_binary: str) -> tuple[str, str]:
    result = subprocess.run(
        [colmap_binary, "feature_extractor", "-h"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    modern = "FeatureExtraction.use_gpu" in (result.stdout + result.stderr)
    if modern:
        return "--FeatureExtraction.use_gpu", "--FeatureMatching.use_gpu"
    return "--SiftExtraction.use_gpu", "--SiftMatching.use_gpu"


def _write_measured_intrinsics(database: Path, frame_index: Path, images_dir: Path) -> int:
    metadata = json.loads(frame_index.read_text(encoding="utf-8"))["images"]
    connection = sqlite3.connect(str(database))
    try:
        rows = connection.execute("SELECT name, camera_id FROM images").fetchall()
        counts: dict[int, int] = {}
        for _, camera_id in rows:
            counts[camera_id] = counts.get(camera_id, 0) + 1
        shared = [camera_id for camera_id, count in counts.items() if count > 1]
        if shared:
            raise RuntimeError(
                "COLMAP assigned a shared camera despite per-image camera configuration"
            )

        updated = 0
        for name, camera_id in rows:
            entry = metadata.get(name)
            if entry is None:
                continue
            image = images_dir / name
            if not image.exists():
                continue
            width, height = entry["image_size"]
            intrinsics = np.asarray(entry["intrinsics"], dtype=np.float64)
            parameters = np.asarray(
                [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]],
                dtype=np.float64,
            )
            connection.execute(
                "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 "
                "WHERE camera_id=?",
                (PINHOLE_MODEL_ID, width, height, parameters.tobytes(), camera_id),
            )
            updated += 1
        connection.commit()
        return updated
    finally:
        connection.close()


def run_sfm(
    images_dir: Path,
    frame_index: Path,
    colmap_dir: Path,
    *,
    masks_dir: Path | None = None,
    colmap_binary: str = "colmap",
    use_gpu: bool = True,
    sequential_overlap: int = 10,
    matcher: str = "sequential",
) -> Path:
    """Build a sparse model and return the largest connected reconstruction."""
    if colmap_dir.exists():
        shutil.rmtree(colmap_dir)
    colmap_dir.mkdir(parents=True)
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir()
    database = colmap_dir / "database.db"
    raw_log = colmap_dir / "colmap.log"
    extraction_gpu, matching_gpu = _gpu_option_names(colmap_binary)
    gpu_value = "1" if use_gpu else "0"

    feature_command = [
        colmap_binary,
        "feature_extractor",
        "--database_path",
        str(database),
        "--image_path",
        str(images_dir),
        "--ImageReader.single_camera",
        "0",
        "--ImageReader.camera_model",
        "PINHOLE",
        extraction_gpu,
        gpu_value,
    ]
    if masks_dir is not None:
        feature_command.extend(["--ImageReader.mask_path", str(masks_dir)])
    run_command(
        feature_command,
        stage="SfM feature extraction",
        raw_log_file=raw_log,
        progress_patterns=[COLMAP_PROGRESS],
    )

    intrinsics_count = _write_measured_intrinsics(database, frame_index, images_dir)
    get_logger().info("SfM intrinsics | wrote measured calibration for %d images", intrinsics_count)

    if matcher == "exhaustive":
        matcher_command = [
            colmap_binary,
            "exhaustive_matcher",
            "--database_path",
            str(database),
            matching_gpu,
            gpu_value,
        ]
    elif matcher == "sequential":
        matcher_command = [
            colmap_binary,
            "sequential_matcher",
            "--database_path",
            str(database),
            "--SequentialMatching.overlap",
            str(sequential_overlap),
            "--SequentialMatching.loop_detection",
            "0",
            matching_gpu,
            gpu_value,
        ]
    else:
        raise ValueError("matcher must be 'sequential' or 'exhaustive'")
    run_command(
        matcher_command,
        stage="SfM feature matching",
        raw_log_file=raw_log,
        progress_patterns=[COLMAP_PROGRESS],
    )

    mapper_command = [
        colmap_binary,
        "mapper",
        "--database_path",
        str(database),
        "--image_path",
        str(images_dir),
        "--output_path",
        str(sparse_dir),
        "--Mapper.ba_refine_focal_length",
        "0",
        "--Mapper.ba_refine_principal_point",
        "0",
        "--Mapper.ba_refine_extra_params",
        "0",
    ]
    run_command(
        mapper_command,
        stage="SfM mapping",
        raw_log_file=raw_log,
        progress_patterns=[COLMAP_PROGRESS],
    )

    import pycolmap

    candidates = sorted(
        path
        for path in sparse_dir.iterdir()
        if (path / "cameras.bin").exists() or (path / "cameras.txt").exists()
    )
    if not candidates:
        raise RuntimeError(f"COLMAP produced no sparse reconstruction; inspect {raw_log}")
    reconstructions = [(pycolmap.Reconstruction(str(path)), path) for path in candidates]
    reconstruction, selected_model = max(reconstructions, key=lambda item: item[0].num_reg_images())
    input_count = len(list(images_dir.glob("*.jpg")))
    registered_count = reconstruction.num_reg_images()
    registration_ratio = registered_count / max(1, input_count)
    get_logger().info(
        "SfM complete | %d/%d images registered (%.0f%%) | %d sparse points | model %s",
        registered_count,
        input_count,
        registration_ratio * 100.0,
        reconstruction.num_points3D(),
        selected_model.name,
    )
    if registered_count < 40 or registration_ratio < 0.60:
        raise RuntimeError(
            f"Sparse reconstruction quality gate failed: {registered_count}/{input_count} images "
            f"registered ({registration_ratio:.0%}). Dense reconstruction was not started."
        )
    return selected_model
