"""Offline MediaPipe detection, pixel geometry, and frame quality checks."""

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


@dataclass(frozen=True)
class Observation:
    span_px: float | None
    quality: str
    landmarks: np.ndarray | None = None
    pose: tuple[float, float, float] | None = None


def pose_angles(matrix: np.ndarray) -> tuple[float, float, float]:
    """Return yaw (Y), pitch (X), roll (Z) in degrees, stripping scale."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Invalid face pose matrix")
    u, singular, vt = np.linalg.svd(matrix[:3, :3])
    if singular.min() < 1e-6 or singular.max() / singular.min() > 1.2:
        raise ValueError("Degenerate face pose matrix")
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        raise ValueError("Reflected face pose matrix")
    yaw = math.asin(float(np.clip(-rotation[2, 0], -1, 1)))
    pitch = math.atan2(rotation[2, 1], rotation[2, 2])
    roll = math.atan2(rotation[1, 0], rotation[0, 0])
    return tuple(math.degrees(a) for a in (yaw, pitch, roll))


def assess(frame: np.ndarray, faces: list, matrices: list) -> Observation:
    if not faces:
        return Observation(None, "No face detected")
    if len(faces) != 1:
        return Observation(None, "Multiple faces - use one person")
    points = np.asarray([(p.x, p.y) for p in faces[0]], dtype=float)
    if (points.ndim != 2 or points.shape[0] < 468 or points.shape[1] != 2
            or not np.isfinite(points).all()):
        return Observation(None, "Invalid facial landmarks")
    height, width = frame.shape[:2]
    pixels = points * np.array([width, height])
    if np.any(points < 0.01) or np.any(points > 0.99):
        return Observation(None, "Face clipped - center your whole face", pixels)
    span = float(np.linalg.norm(pixels[33] - pixels[263]))
    if span < 40:
        return Observation(None, "Face too small - move closer", pixels)
    if not matrices:
        return Observation(None, "Face pose unavailable", pixels)
    try:
        pose = pose_angles(matrices[0])
    except (ValueError, np.linalg.LinAlgError):
        return Observation(None, "Invalid face pose", pixels)
    yaw, pitch, roll = pose
    if abs(yaw) > 20 or abs(pitch) > 20 or abs(roll) > 25:
        return Observation(None, "Face camera directly - straighten head", pixels, pose)
    x1, y1 = np.floor(pixels.min(axis=0)).astype(int)
    x2, y2 = np.ceil(pixels.max(axis=0)).astype(int)
    roi = frame[y1:y2 + 1, x1:x2 + 1]
    if roi.size == 0:
        return Observation(None, "Invalid face region", pixels, pose)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    if brightness < 35:
        return Observation(None, "Face too dark - improve lighting", pixels, pose)
    if brightness > 240:
        return Observation(None, "Face overexposed - reduce lighting", pixels, pose)
    if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < 40:
        return Observation(None, "Face blurred - hold still or focus camera", pixels, pose)
    return Observation(span, "OK", pixels, pose)


class FaceDetector:
    def __init__(self, model_path: Path):
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing model: {model_path}. Rebuild the Docker image.")
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path), delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=2,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            output_facial_transformation_matrixes=True)
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self.last_timestamp = -1

    def detect(self, frame: np.ndarray, timestamp_ms: int) -> Observation:
        # MediaPipe VIDEO requires strictly increasing timestamps, even at high FPS.
        timestamp_ms = max(timestamp_ms, self.last_timestamp + 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        self.last_timestamp = timestamp_ms
        return assess(frame, result.face_landmarks, result.facial_transformation_matrixes)

    def close(self) -> None:
        self.landmarker.close()

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def draw_landmarks(frame: np.ndarray, observation: Observation) -> np.ndarray:
    display = frame.copy()
    if observation.landmarks is None:
        return display
    points = observation.landmarks
    color = (100, 220, 100) if observation.quality == "OK" else (40, 160, 240)
    for x, y in points:
        cv2.circle(display, (round(x), round(y)), 1, color, -1, cv2.LINE_AA)
    for connection in mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS:
        a, b = points[connection.start], points[connection.end]
        cv2.line(display, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)),
                 color, 1, cv2.LINE_AA)
    cv2.line(display, tuple(np.rint(points[33]).astype(int)),
             tuple(np.rint(points[263]).astype(int)), (255, 200, 60), 2, cv2.LINE_AA)
    return display
