import argparse
import signal
from unittest.mock import Mock

import csv
import numpy as np
import pytest

from farfinder.app import positive_int, run
from farfinder.storage import DataStore
from farfinder.vision import Observation


def test_no_display_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    with pytest.raises(RuntimeError, match="DISPLAY is missing"):
        run(argparse.Namespace(data_dir=tmp_path))


def test_camera_open_failure_releases_resources(tmp_path, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    camera = Mock()
    camera.isOpened.return_value = False
    monkeypatch.setattr("farfinder.app.cv2.VideoCapture", lambda *args: camera)
    cleanup = Mock()
    monkeypatch.setattr("farfinder.app.cv2.destroyAllWindows", cleanup)
    handler = signal.getsignal(signal.SIGTERM)
    with pytest.raises(RuntimeError, match="Cannot open"):
        run(argparse.Namespace(camera="/dev/video0", data_dir=tmp_path))
    camera.release.assert_called_once()
    cleanup.assert_called_once()
    assert signal.getsignal(signal.SIGTERM) == handler


def test_keyboard_calibration_completes_saves_and_logs_measurement(tmp_path, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    camera = Mock()
    camera.isOpened.return_value = True
    camera.read.return_value = (True, frame)
    monkeypatch.setattr("farfinder.app.cv2.VideoCapture", lambda *args: camera)
    tick = [0.0]
    monkeypatch.setattr("farfinder.app.time.monotonic", lambda: tick[0])
    index = [0]
    def detect(frame, timestamp):
        span = 200 if index[0] <= 45 else 100
        index[0] += 1
        tick[0] += 0.04
        return Observation(span, "OK")
    detector = Mock()
    detector.detect.side_effect = detect
    detector.__enter__ = Mock(return_value=detector)
    detector.__exit__ = Mock(return_value=False)
    monkeypatch.setattr("farfinder.app.FaceDetector", lambda path: detector)
    keys = iter([ord("1")] + [-1] * 44 + [ord("2")] + [-1] * 44 + [ord("q")])
    monkeypatch.setattr("farfinder.app.cv2.waitKey", lambda delay: next(keys))
    monkeypatch.setattr("farfinder.app.cv2.getWindowProperty", lambda *args: 1)
    for method in ("namedWindow", "resizeWindow", "imshow", "destroyAllWindows"):
        monkeypatch.setattr(f"farfinder.app.cv2.{method}", Mock())
    args = argparse.Namespace(camera="/dev/video0", width=640, height=480,
                              fps=30, model=tmp_path / "model.task", data_dir=tmp_path)
    assert run(args) == 0
    calibration = DataStore(tmp_path).load("/dev/video0", 640, 480)
    assert calibration.estimate(100) == pytest.approx(60)
    with (tmp_path / "calibration.csv").open() as file:
        assert len(list(csv.DictReader(file))) == 2
    with next(tmp_path.glob("measurements_*.csv")).open() as file:
        rows = list(csv.DictReader(file))
    assert any(row["quality"].startswith("Calibration in progress") for row in rows)
    assert any(float(row["distance_cm"] or 0) == pytest.approx(60) for row in rows)
    camera.release.assert_called_once()


@pytest.mark.parametrize("value", ["0", "-1"])
def test_invalid_camera_dimensions(value):
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int(value)
