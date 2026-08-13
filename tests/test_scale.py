import numpy as np

from poc.pipeline.scale import _ransac_similarity, umeyama_similarity


def test_umeyama_recovers_similarity() -> None:
    source = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target = 2.5 * (source @ rotation.T) + np.asarray([3.0, -2.0, 7.0])
    scale, recovered_rotation, translation = umeyama_similarity(source, target)
    np.testing.assert_allclose(scale, 2.5, rtol=1e-10)
    np.testing.assert_allclose(recovered_rotation, rotation, atol=1e-10)
    np.testing.assert_allclose(translation, [3.0, -2.0, 7.0], atol=1e-10)


def test_ransac_rejects_trajectory_outliers() -> None:
    rng = np.random.default_rng(11)
    source = rng.normal(size=(80, 3))
    target = 0.42 * source + np.asarray([1.0, 2.0, 3.0])
    target += rng.normal(scale=0.001, size=target.shape)
    target[[4, 17, 55]] += 0.25
    scale, _, _, inliers = _ransac_similarity(source, target, threshold_m=0.01)
    np.testing.assert_allclose(scale, 0.42, rtol=0.01)
    assert inliers.sum() >= 75
    assert not inliers[[4, 17, 55]].any()
