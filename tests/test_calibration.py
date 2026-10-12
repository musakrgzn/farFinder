import math

import pytest

from farfinder.calibration import Calibration, CalibrationError, StableCapture


def calibrated():
    return Calibration.fit(200, 100, "/dev/video0", 640, 480)


def test_two_point_fit_and_interpolation():
    model = calibrated()
    assert model.estimate(200) == pytest.approx(30)
    assert model.estimate(100) == pytest.approx(60)
    assert model.estimate(6000 / 45) == pytest.approx(45)


def test_fit_accounts_for_nonzero_offset():
    model = Calibration.fit(250, 100, "/dev/video0", 640, 480)
    assert model.offset_cm == pytest.approx(10)
    assert model.estimate(5000 / 35) == pytest.approx(45)


@pytest.mark.parametrize("near,far", [(100, 200), (100, 100), (0, 100),
                                     (200, 39), (math.nan, 100), (200, math.inf),
                                     (500, 100), (150, 100)])
def test_rejects_bad_or_implausible_calibration(near, far):
    with pytest.raises(CalibrationError):
        Calibration.fit(near, far, "/dev/video0", 640, 480)


def test_payload_and_setup_validation():
    model = calibrated()
    assert Calibration.from_dict(model.to_dict()) == model
    for key, value in [("scale", math.inf), ("offset_cm", 99), ("width", "640"),
                       ("feature", "wrong"), ("schema_version", 2), ("created_at", "bad")]:
        payload = model.to_dict() | {key: value}
        with pytest.raises(CalibrationError):
            Calibration.from_dict(payload)
    with pytest.raises(CalibrationError):
        model.check_setup("/dev/video0", 1280, 720)
    with pytest.raises(CalibrationError):
        model.check_setup("/dev/video1", 640, 480)


@pytest.mark.parametrize("span", [0, -1, math.nan, math.inf])
def test_estimate_rejects_invalid_span(span):
    with pytest.raises(CalibrationError):
        calibrated().estimate(span)


def test_capture_requires_sample_count_and_duration():
    capture = StableCapture(30, 0, sample_count=3, min_seconds=1)
    assert capture.add(200, 0) is None
    assert capture.add(201, 0.2) is None
    assert capture.add(199, 0.4) is None
    result = capture.add(200, 1)
    assert result.target_cm == 30
    assert result.span_px == pytest.approx(200)
    assert result.sample_count == 4
    assert result.relative_std < 0.03


def test_bad_frame_clears_capture_and_movement_starts_new_batch():
    capture = StableCapture(30, 0, sample_count=3, min_seconds=1)
    capture.add(200, 0)
    capture.add(200, 0.5)
    assert capture.add(None, 0.6) is None
    assert capture.samples == []
    capture.add(200, 0.7)
    assert capture.add(250, 0.8) is None
    assert len(capture.samples) == 1
    capture.add(250, 1.3)
    assert capture.add(250, 1.9).span_px == 250


def test_noisy_capture_and_timeout():
    capture = StableCapture(60, 0, sample_count=3, min_seconds=1)
    capture.add(100, 0)
    capture.add(107, 0.5)
    assert capture.add(100, 1) is None
    assert capture.samples == []
    with pytest.raises(CalibrationError, match="timed out"):
        capture.add(100, 15)
