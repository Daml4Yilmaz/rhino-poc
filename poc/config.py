"""Typed reconstruction configuration and case-directory layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReconstructionConfig:
    output_dir: Path
    frame_count: int = 120
    minimum_sharpness: float = 35.0
    frame_max_dimension: int = 1400
    rotation: str = "clockwise"
    sequential_overlap: int = 10
    mvs_reference_count: int = 96
    mvs_source_images: int = 6
    mvs_geometric_consistency: bool = False
    mvs_cache_gb: int | None = None
    mvs_max_image_size: int | None = None
    poisson_depth: int = 9
    texture_scale_factor: float = 1.0
    scale_agreement_percent: float = 2.0
    colmap_binary: str = "colmap"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "case.json"

    @property
    def log_path(self) -> Path:
        return self.output_dir / "run.log"

    @property
    def images_dir(self) -> Path:
        return self.output_dir / "images"

    @property
    def frame_index_path(self) -> Path:
        return self.output_dir / "frames.json"

    @property
    def capture_quality_path(self) -> Path:
        return self.output_dir / "capture_quality.json"

    @property
    def masks_dir(self) -> Path:
        return self.output_dir / "masks"

    @property
    def reconstruction_images_dir(self) -> Path:
        return self.output_dir / "reconstruction_images"

    @property
    def colmap_dir(self) -> Path:
        return self.output_dir / "colmap"

    @property
    def sparse_dir(self) -> Path:
        return self.colmap_dir / "sparse"

    @property
    def sfm_metrics_path(self) -> Path:
        return self.output_dir / "sfm.json"

    @property
    def dense_dir(self) -> Path:
        return self.colmap_dir / "dense"

    @property
    def raw_mesh_path(self) -> Path:
        return self.output_dir / "face_mesh_raw.ply"

    @property
    def scale_path(self) -> Path:
        return self.output_dir / "scale.json"

    @property
    def authoritative_mesh_path(self) -> Path:
        return self.output_dir / "face_geometry.ply"

    @property
    def geometry_metadata_path(self) -> Path:
        return self.output_dir / "geometry.json"

    @property
    def texture_dir(self) -> Path:
        return self.output_dir / "texture"

    @property
    def texture_workspace_dir(self) -> Path:
        return self.colmap_dir / "texture_workspace"

    @property
    def textured_mesh_path(self) -> Path:
        return self.texture_dir / "mesh.ply"

    @property
    def texture_image_path(self) -> Path:
        return self.texture_dir / "texture.png"

    @property
    def glb_path(self) -> Path:
        return self.output_dir / "face_model.glb"

    @property
    def landmarks_path(self) -> Path:
        return self.output_dir / "landmarks.json"

    @property
    def measurements_path(self) -> Path:
        return self.output_dir / "measurements.json"

    @property
    def quality_report_path(self) -> Path:
        return self.output_dir / "quality_report.json"

    @property
    def quality_report_html_path(self) -> Path:
        return self.output_dir / "quality_report.html"
