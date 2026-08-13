import math

from poc.pipeline.measure import compute_measurements


def test_measurements_are_finite_and_have_expected_distances() -> None:
    landmarks = {
        "glabella": [0.0, 10.0, 0.0],
        "nasion": [0.0, 0.0, 0.0],
        "pronasale": [0.0, 0.0, 50.0],
        "subnasale": [0.0, -20.0, 20.0],
        "columella": [0.0, -18.0, 24.0],
        "labiale_superius": [0.0, -27.0, 18.0],
        "left_alare": [-17.5, -14.0, 15.0],
        "right_alare": [17.5, -14.0, 15.0],
        "left_endocanthion": [-15.0, 10.0, 0.0],
        "right_endocanthion": [15.0, 10.0, 0.0],
    }
    measurements = compute_measurements(landmarks)
    assert measurements["nose_length_mm"] == 50.0
    assert measurements["nose_width_mm"] == 35.0
    assert measurements["midline_deviation_mm"] == 0.0
    assert all(math.isfinite(value) for value in measurements.values())
