from pathlib import Path

import numpy as np
import pytest

from poc.diagnostics.poisson import (
    PoissonDiagnosticConfig,
    _normal_statistics,
    _surface_thickness_statistics,
)


def test_config_rejects_unrequested_poisson_depth() -> None:
    with pytest.raises(ValueError, match="7, 8, and 9"):
        PoissonDiagnosticConfig(poisson_depths=(10,), production_poisson_depth=10).validate()


def test_config_requires_production_depth_in_experiment() -> None:
    with pytest.raises(ValueError, match="include production_poisson_depth"):
        PoissonDiagnosticConfig(poisson_depths=(7, 8), production_poisson_depth=9).validate()


def test_normal_statistics_detect_a_reversed_neighbor() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]])
    normals = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])

    result = _normal_statistics(
        points,
        normals,
        scale_mm_per_unit=1.0,
        sample_count=4,
        neighbor_count=1,
        random_seed=7,
    )

    assert result["status"] == "available"
    assert result["fraction_dot_lt_0"] > 0
    assert result["fraction_dot_lt_minus_0_9"] > 0


def test_planar_surface_has_negligible_local_plane_residual() -> None:
    coordinates = np.linspace(-1.0, 1.0, 15)
    xx, yy = np.meshgrid(coordinates, coordinates)
    points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])

    result = _surface_thickness_statistics(
        points,
        np.ones(len(points), dtype=bool),
        scale_mm_per_unit=1.0,
        radius_mm=0.5,
        max_nn=32,
        min_neighbors=5,
        sample_count=100,
        random_seed=11,
    )

    assert result["status"] == "available"
    assert result["absolute_local_plane_residual_mm"]["p99"] < 1e-9


def test_diagnostic_entry_point_requires_permanent_fused_artifact(tmp_path: Path) -> None:
    from poc.diagnostics.poisson import run_poisson_diagnostic

    with pytest.raises(FileNotFoundError, match="stereo-fusion artifact"):
        run_poisson_diagnostic(tmp_path / "case", tmp_path / "diagnostics")
