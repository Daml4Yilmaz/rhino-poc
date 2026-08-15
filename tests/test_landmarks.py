import numpy as np

from poc.pipeline.landmarks import _robust_surface_consensus


def test_surface_consensus_rejects_distant_observations() -> None:
    rng = np.random.default_rng(7)
    cluster = rng.normal(loc=[0.0, 0.0, 0.0], scale=0.0003, size=(30, 3))
    outliers = np.asarray([[0.020, 0.0, 0.0], [-0.020, 0.0, 0.0]])
    point, inliers, distances = _robust_surface_consensus(np.vstack([cluster, outliers]))

    assert inliers.sum() == len(cluster)
    assert np.linalg.norm(point) < 0.0002
    assert np.min(distances[~inliers]) > 0.019
