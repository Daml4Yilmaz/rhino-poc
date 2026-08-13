"""Build a view-selected color texture atlas with COLMAP."""

from __future__ import annotations

import shutil
from pathlib import Path

from poc.logging_utils import get_logger, run_command


def run_texture_mapping(
    dense_workspace: Path,
    raw_mesh_path: Path,
    output_dir: Path,
    *,
    colmap_binary: str = "colmap",
    texture_scale_factor: float = 1.0,
) -> tuple[Path, Path]:
    """Project registered source photographs into a UV texture atlas."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    run_command(
        [
            colmap_binary,
            "mesh_texturer",
            "--workspace_path",
            str(dense_workspace),
            "--input_path",
            str(raw_mesh_path),
            "--output_path",
            str(output_dir),
            "--MeshTextureMapping.apply_color_correction",
            "1",
            "--MeshTextureMapping.texture_scale_factor",
            str(texture_scale_factor),
        ],
        stage="Texture atlas",
        raw_log_file=dense_workspace.parent / "colmap.log",
    )
    textured_mesh = output_dir / "mesh.ply"
    texture_image = output_dir / "texture.png"
    if not textured_mesh.is_file() or not texture_image.is_file():
        raise RuntimeError("COLMAP texture mapping did not produce mesh.ply and texture.png")
    get_logger().info(
        "Texture atlas complete | %s | %.1f MB",
        texture_image,
        texture_image.stat().st_size / 1_000_000.0,
    )
    return textured_mesh, texture_image
