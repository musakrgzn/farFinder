import csv
import json

import pytest

from farfinder.calibration import Calibration, CalibrationError, CaptureResult
from farfinder.measurement import Reading
from farfinder.storage import DataStore, MeasurementLogger


def save(store, near=200, far=100):
    calibration = Calibration.fit(near, far, "/dev/video0", 640, 480)
    store.save(calibration, [CaptureResult(30, near, 45, 0.01),
                             CaptureResult(60, far, 45, 0.01)])
    return calibration


def test_persistence_reload_and_csv_history(tmp_path):
    store = DataStore(tmp_path)
    assert store.load("/dev/video0", 640, 480) is None
    first = save(store)
    assert store.load("/dev/video0", 640, 480) == first
    latest = save(store, 220, 110)
    assert store.load("/dev/video0", 640, 480) == latest
    with (tmp_path / "calibration.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 4
    assert [int(row["target_cm"]) for row in rows] == [30, 60, 30, 60]
    assert rows[-1]["calibration_id"] == latest.calibration_id
    store.reset()
    assert store.load("/dev/video0", 640, 480) is None
    assert (tmp_path / "calibration.csv").exists()


@pytest.mark.parametrize("content", ["bad-json", "[]", "{}", '{"scale": "bad"}'])
def test_corrupt_calibration_is_rejected(tmp_path, content):
    store = DataStore(tmp_path)
    store.calibration_path.write_text(content)
    with pytest.raises(CalibrationError):
        store.load("/dev/video0", 640, 480)


def test_failed_atomic_replace_preserves_previous_calibration(tmp_path, monkeypatch):
    store = DataStore(tmp_path)
    previous = save(store)
    def fail_replace(source, target):
        raise OSError("simulated write failure")
    monkeypatch.setattr("farfinder.storage.os.replace", fail_replace)
    with pytest.raises(OSError):
        save(store, 220, 110)
    assert store.load("/dev/video0", 640, 480) == previous
    assert not list(tmp_path.glob(".calibration-*.tmp"))
    assert json.loads(store.calibration_path.read_text())["calibration_id"] == previous.calibration_id


def test_measurement_csv_has_blanks_on_rejection_and_respects_rate(tmp_path):
    calibration = Calibration.fit(200, 100, "/dev/video0", 640, 480)
    with MeasurementLogger(tmp_path) as logger:
        start = logger.started
        logger.write(start, Reading(45, 45, "OK", True), 6000 / 45, 30, calibration, (0, 0, 0))
        logger.write(start + 0.01, Reading(45, 45, "OK", True), 6000 / 45, 30, calibration)
        logger.write(start + 1, Reading(None, None, "No face detected"), None, 30, calibration)
        path = logger.path
    with path.open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert float(rows[0]["distance_cm"]) == 45
    assert rows[1]["distance_cm"] == rows[1]["raw_distance_cm"] == ""
    assert rows[1]["quality"] == "No face detected"
    assert rows[1]["calibrated"] == "1"
