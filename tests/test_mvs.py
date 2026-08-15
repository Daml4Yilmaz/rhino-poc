from pathlib import Path

import numpy as np

from poc.config import ReconstructionConfig
from poc.pipeline.mvs import _point_cloud_diagnostics


class _PointCloudStub:
    def __init__(self) -> None:
        self.points = np.asarray(
            [[-1.0, 2.0, 3.0], [2.0, 6.0, 8.0], [0.5, 4.0, 5.0]], dtype=np.float64
        )

    @staticmethod
    def has_normals() -> bool:
        return True

    @staticmethod
    def has_colors() -> bool:
        return True


def test_fused_point_cloud_is_a_permanent_case_output(tmp_path: Path) -> None:
    config = ReconstructionConfig(output_dir=tmp_path / "case")

    assert config.fused_point_cloud_path.parent == config.output_dir
    assert config.fused_point_cloud_path.parent != config.dense_dir
    assert config.mvs_metrics_path.parent == config.output_dir


def test_pre_poisson_point_cloud_diagnostics(tmp_path: Path) -> None:
    artifact = tmp_path / "face_dense_fused.ply"
    artifact.write_bytes(b"direct stereo fusion output")

    result = _point_cloud_diagnostics(_PointCloudStub(), artifact, scale_mm_per_unit=10.0)

    assert result["role"] == "raw_fused_dense_point_cloud_pre_poisson"
    assert result["generated_by"] == "COLMAP stereo_fusion"
    assert result["geometry_state"].endswith("before_normal_estimation_and_poisson")
    assert result["path"] == artifact.name
    assert result["point_count"] == 3
    assert result["normals_present_in_persisted_artifact"] is True
    assert result["colors_present_in_persisted_artifact"] is True
    assert result["bounding_box_min"] == [-1.0, 2.0, 3.0]
    assert result["bounding_box_max"] == [2.0, 6.0, 8.0]
    assert result["metric_bounding_box_extent_mm"] == [30.0, 40.0, 50.0]
    assert result["file_size_bytes"] == artifact.stat().st_size
    assert len(result["sha256"]) == 64
