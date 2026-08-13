import numpy as np

from poc.pipeline.frames import rotation_transform, transform_intrinsics


def test_clockwise_rotation_maps_source_corners() -> None:
    transform = rotation_transform((1920, 1440), "clockwise")
    assert transform.output_size == (1440, 1920)
    top_left = transform.source_to_image @ np.asarray([0.0, 0.0, 1.0])
    bottom_right = transform.source_to_image @ np.asarray([1919.0, 1439.0, 1.0])
    np.testing.assert_allclose(top_left, [1439.0, 0.0, 1.0])
    np.testing.assert_allclose(bottom_right, [0.0, 1919.0, 1.0])


def test_clockwise_intrinsics_remain_a_conventional_pinhole_matrix() -> None:
    source = np.asarray([[1370.0, 0.0, 963.0], [0.0, 1368.0, 722.0], [0.0, 0.0, 1.0]])
    transformed = transform_intrinsics(source, (1920, 1440), "clockwise", 0.5)
    np.testing.assert_allclose(
        transformed,
        [[684.0, 0.0, 358.5], [0.0, 685.0, 481.5], [0.0, 0.0, 1.0]],
    )


def test_counterclockwise_intrinsics() -> None:
    source = np.asarray([[1370.0, 0.0, 963.0], [0.0, 1368.0, 722.0], [0.0, 0.0, 1.0]])
    transformed = transform_intrinsics(source, (1920, 1440), "counterclockwise", 1.0)
    np.testing.assert_allclose(
        transformed,
        [[1368.0, 0.0, 722.0], [0.0, 1370.0, 956.0], [0.0, 0.0, 1.0]],
    )
