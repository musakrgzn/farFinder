import math

import pytest

from farfinder.calibration import Calibration
from farfinder.measurement import DistanceSmoother, MeasurementEngine


def engine():
    return MeasurementEngine(Calibration.fit(200, 100, "/dev/video0", 640, 480))


def test_smoothing_rejects_isolated_spike_then_tracks_movement():
    smoother = DistanceSmoother()
    for value in [45] * 5:
        assert smoother.update(value) == 45
    assert smoother.update(110) == 45
    for value in [60] * 15:
        result = smoother.update(value)
    assert result == pytest.approx(60, abs=0.1)


@pytest.mark.parametrize("quality", ["No face detected", "Multiple faces", "Face blurred"])
def test_bad_frame_clears_distance_and_filter_history(quality):
    measurement = engine()
    assert measurement.process(200).distance_cm == 30
    rejected = measurement.process(None, quality)
    assert rejected.distance_cm is None
    assert rejected.raw_distance_cm is None
    assert measurement.process(100).distance_cm == pytest.approx(60)


def test_uncalibrated_and_reset():
    measurement = MeasurementEngine()
    assert measurement.process(100).distance_cm is None
    measurement.set_calibration(engine().calibration)
    assert measurement.process(100).distance_cm == pytest.approx(60)
    measurement.set_calibration(None)
    assert measurement.process(100).distance_cm is None


def test_extrapolation_is_labeled_and_extreme_distances_are_rejected():
    measurement = engine()
    reading = measurement.process(80)
    assert reading.raw_distance_cm == pytest.approx(75)
    assert not reading.within_calibrated_range
    assert "Extrapolated" in reading.quality
    assert measurement.process(400).distance_cm is None
    assert measurement.process(45).distance_cm is None


@pytest.mark.parametrize("span", [math.nan, math.inf, -1, 0, 39])
def test_invalid_landmark_spans_cannot_produce_distance(span):
    assert engine().process(span).distance_cm is None
