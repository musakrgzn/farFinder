from pathlib import Path
from types import SimpleNamespace
import os

import cv2
import numpy as np
import pytest

from farfinder.vision import FaceDetector, assess, draw_landmarks, pose_angles


def face():
    points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
    points[33] = SimpleNamespace(x=0.35, y=0.4)
    points[263] = SimpleNamespace(x=0.65, y=0.4)
    points[10] = SimpleNamespace(x=0.3, y=0.25)
    points[152] = SimpleNamespace(x=0.7, y=0.75)
    return points


def textured_frame():
    y, x = np.indices((480, 640))
    gray = ((x + y) % 2 * 80 + 80).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def test_quality_accepts_one_frontal_face_and_uses_pixel_dimensions():
    observation = assess(textured_frame(), [face()], [np.eye(4)])
    assert observation.quality == "OK"
    assert observation.span_px == pytest.approx(192)
    assert observation.pose == pytest.approx((0, 0, 0))


def test_preview_draws_landmarks_without_changing_inference_frame():
    frame = textured_frame()
    original = frame.copy()
    observation = assess(frame, [face()], [np.eye(4)])
    preview = draw_landmarks(frame, observation)
    assert preview.shape == frame.shape
    assert not np.array_equal(preview, original)
    assert np.array_equal(frame, original)


def test_missing_multiple_clipped_and_nonfinite_faces_are_rejected():
    frame = textured_frame()
    assert "No face" in assess(frame, [], []).quality
    assert "Multiple faces" in assess(frame, [face(), face()], []).quality
    clipped = face()
    clipped[10].x = -0.1
    assert "clipped" in assess(frame, [clipped], [np.eye(4)]).quality
    clipped[10].x = np.nan
    assert "Invalid facial landmarks" in assess(frame, [clipped], [np.eye(4)]).quality
    assert "pose unavailable" in assess(frame, [face()], []).quality


@pytest.mark.parametrize("level,reason", [(0, "dark"), (255, "overexposed"), (128, "blurred")])
def test_bad_lighting_and_blur_rejected(level, reason):
    frame = np.full((480, 640, 3), level, dtype=np.uint8)
    observation = assess(frame, [face()], [np.eye(4)])
    assert reason in observation.quality
    assert observation.span_px is None


@pytest.mark.parametrize("axis,index", [("yaw", 0), ("pitch", 1), ("roll", 2)])
def test_pose_angles_and_turned_heads_rejected(axis, index):
    theta = np.radians(30)
    c, s = np.cos(theta), np.sin(theta)
    rotations = {"yaw": [[c, 0, s], [0, 1, 0], [-s, 0, c]],
                 "pitch": [[1, 0, 0], [0, c, -s], [0, s, c]],
                 "roll": [[c, -s, 0], [s, c, 0], [0, 0, 1]]}
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(rotations[axis]) * 1.05
    assert pose_angles(matrix)[index] == pytest.approx(30)
    assert "Face camera directly" in assess(textured_frame(), [face()], [matrix]).quality


@pytest.mark.parametrize("matrix", [np.zeros((4, 4)), np.eye(3), np.full((4, 4), np.nan)])
def test_invalid_pose_matrices(matrix):
    with pytest.raises(ValueError):
        pose_angles(matrix)


def test_model_missing_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Rebuild"):
        FaceDetector(tmp_path / "missing.task")


MODEL = Path(os.environ.get("FARFINDER_MODEL_PATH", "/app/models/face_landmarker.task"))


@pytest.mark.skipif(not MODEL.is_file(), reason="Bundled model smoke test requires the Docker image")
def test_bundled_model_runs_offline_and_accepts_repeated_input_timestamps():
    # Real native MediaPipe inference, not a mocked detector; needs no camera/display.
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with FaceDetector(MODEL) as detector:
        assert detector.detect(frame, 0).quality == "No face detected"
        assert detector.detect(frame, 0).quality == "No face detected"
        assert detector.last_timestamp == 1


def test_opencv_has_gui_support():
    assert "GUI:                           NONE" not in cv2.getBuildInformation()
    assert "QT" in cv2.getBuildInformation()
