"""Webcam window, keyboard calibration workflow, and resource lifecycle."""

import argparse
from collections import deque
import logging
import os
from pathlib import Path
import signal
import textwrap
import time

import cv2
import numpy as np

from farfinder.calibration import Calibration, CalibrationError, CaptureResult, StableCapture
from farfinder.measurement import MeasurementEngine, Reading
from farfinder.storage import DataStore, MeasurementLogger
from farfinder.vision import FaceDetector, Observation, draw_landmarks

LOG = logging.getLogger(__name__)
WINDOW = "farFinder - webcam-to-face distance"


def panel(frame: np.ndarray, observation: Observation, reading: Reading, fps: float,
          calibration: Calibration | None, message: str, capture: StableCapture | None,
          has_near_point: bool) -> np.ndarray:
    video = cv2.flip(draw_landmarks(frame, observation), 1)
    height, width = video.shape[:2]
    canvas = np.full((max(height, 560), width + 400, 3), (28, 24, 22), dtype=np.uint8)
    canvas[:height, :width] = video
    x = width + 20
    def label(text: str, y: int, scale: float = 0.55, color=(230, 230, 230)) -> None:
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, 1, cv2.LINE_AA)
    label("farFinder", 35, 0.95, (255, 210, 100))
    distance = "-- cm" if reading.distance_cm is None else f"{reading.distance_cm:.1f} cm"
    label(distance, 92, 1.3, (100, 230, 150) if reading.distance_cm is not None else (120, 180, 250))
    label(f"FPS: {fps:.1f}", 127)
    state = "READY" if calibration else "NEEDS CALIBRATION"
    if capture:
        state = f"CAPTURING {capture.target_cm} CM"
    elif has_near_point:
        state = "30 CM SAVED - PRESS 2 AT 60 CM"
    label(f"Calibration: {state}", 160, 0.47)
    label("Quality", 194, color=(255, 210, 100))
    for i, line in enumerate(textwrap.wrap(reading.quality, 38)):
        label(line, 219 + i * 22, 0.49)
    if observation.pose:
        yaw, pitch, roll = observation.pose
        label(f"Yaw {yaw:.0f} / pitch {pitch:.0f} / roll {roll:.0f} deg", 274, 0.47)
    if observation.span_px:
        label(f"Outer eye span: {observation.span_px:.1f} px", 299, 0.5)
    for i, line in enumerate(textwrap.wrap(capture.progress if capture else message, 39)):
        label(line, 335 + i * 22, 0.48, (120, 200, 255))
    label("1: capture 30 cm (start calibration)", 438, 0.49)
    label("2: capture 60 cm (finish calibration)", 465, 0.49)
    label("R: reset calibration     Q / Esc: quit", 492, 0.49)
    label("CSV recording / local CPU / offline", 533, 0.46, (160, 160, 160))
    return canvas


def run(args: argparse.Namespace) -> int:
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("DISPLAY is missing. Start from an Ubuntu X11/XWayland desktop; see README.")
    store = DataStore(args.data_dir)
    camera = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    previous_handler = signal.signal(signal.SIGTERM, stop)
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Cannot open {args.camera}. Check the device, VIDEO_GID, and other camera apps.")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        camera.set(cv2.CAP_PROP_FPS, args.fps)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = camera.read()
        if not ok or frame is None:
            raise RuntimeError("Webcam opened but delivered no frame. Check camera format and permissions.")
        height, width = frame.shape[:2]
        LOG.info("Camera %s: actual frame size %dx%d", args.camera, width, height)
        message = "Measure 30 cm from camera lens to nose, then press 1."
        try:
            calibration = store.load(args.camera, width, height)
        except CalibrationError as exc:
            calibration = None
            message = str(exc)
            LOG.warning("%s", exc)
        engine = MeasurementEngine(calibration)
        if calibration:
            message = "Saved calibration loaded. Use the same person and camera."
        near: CaptureResult | None = None
        capture: StableCapture | None = None
        times: deque[float] = deque(maxlen=30)
        started = time.monotonic()
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, min(width + 400, 1280), min(max(height, 560), 800))
        with FaceDetector(args.model) as detector, MeasurementLogger(store.directory) as logger:
            LOG.info("Recording measurements to %s", logger.path)
            while not stopping:
                current_height, current_width = frame.shape[:2]
                if (current_width, current_height) != (width, height):
                    width, height = current_width, current_height
                    engine.set_calibration(None)
                    near, capture = None, None
                    message = "Camera frame size changed. Recalibrate with 1 and 2."
                observation = detector.detect(frame, int((time.monotonic() - started) * 1000))
                now = time.monotonic()
                times.append(now)
                fps = ((len(times) - 1) / (times[-1] - times[0])
                       if len(times) > 1 and times[-1] > times[0] else 0.0)
                if capture:
                    try:
                        completed = capture.add(observation.span_px, now)
                        if completed:
                            capture = None
                            if completed.target_cm == 30:
                                near = completed
                                message = "30 cm captured. Move to 60 cm, then press 2."
                            else:
                                fitted = Calibration.fit(near.span_px, completed.span_px,
                                                         args.camera, width, height)
                                store.save(fitted, [near, completed])
                                engine.set_calibration(fitted)
                                near = None
                                message = "Calibration saved. Distance is now in centimeters."
                                LOG.info("Calibration saved: scale=%.3f offset=%.3f cm",
                                         fitted.scale, fitted.offset_cm)
                    except CalibrationError as exc:
                        capture = None
                        message = str(exc)
                        LOG.warning("%s", exc)
                if capture or near:
                    reading = engine.process(None, "Calibration in progress - hold position")
                else:
                    reading = engine.process(observation.span_px, observation.quality)
                logger.write(now, reading, observation.span_px, fps, engine.calibration, observation.pose)
                cv2.imshow(WINDOW, panel(frame, observation, reading, fps, engine.calibration,
                                         message, capture, near is not None))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("1"):
                    near = None
                    capture = StableCapture(30, time.monotonic())
                    engine.smoother.reset()
                    message = "Hold still at 30 cm."
                elif key == ord("2"):
                    if near is None:
                        message = "First capture 30 cm with 1."
                    else:
                        capture = StableCapture(60, time.monotonic())
                        message = "Hold still at 60 cm."
                elif key in (ord("r"), ord("R")):
                    store.reset()
                    engine.set_calibration(None)
                    near, capture = None, None
                    message = "Calibration reset. Press 1 at 30 cm. CSV history kept."
                if not stopping:
                    ok, frame = camera.read()
                    if not ok or frame is None:
                        raise RuntimeError("Webcam stopped delivering frames. Reconnect it and restart.")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        signal.signal(signal.SIGTERM, previous_handler)
    return 0


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline, two-point calibrated webcam-to-face distance.")
    parser.add_argument("--camera", default="/dev/video0", help="V4L2 device path (default: /dev/video0)")
    parser.add_argument("--width", type=positive_int, default=640)
    parser.add_argument("--height", type=positive_int, default=480)
    parser.add_argument("--fps", type=positive_int, default=30, help="Requested camera FPS")
    parser.add_argument("--model", type=Path,
                        default=Path(os.environ.get("FARFINDER_MODEL_PATH", "/app/models/face_landmarker.task")))
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--head", action="store_true", help="Open automatic head capture / 3D measurement mode")
    parser.add_argument("--capture-mode", choices=["moving-camera", "head-rotation"], default="moving-camera",
                        help="Head capture: stationary subject/camera orbit, or experimental fixed-webcam head turns")
    parser.add_argument("--intrinsics", type=Path, help="Head mode intrinsic calibration JSON")
    parser.add_argument("--reference-square-cm", type=float, help="Independently measured static ChArUco square size")
    parser.add_argument("--reference-uncertainty-cm", type=float, help="Uncertainty of the physical reference dimension")
    parser.add_argument("--ear-model", type=Path, help="Optional local custom ear keypoint ONNX model")
    parser.add_argument("--ear-manifest", type=Path, help="Ear model semantics, checksum and evaluation manifest")
    args = parser.parse_args()
    if (args.reference_square_cm is None) != (args.reference_uncertainty_cm is None):
        parser.error("Supply both physical reference dimension and its uncertainty")
    if (args.ear_model is None) != (args.ear_manifest is None):
        parser.error("Ear model and manifest must be supplied together")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.head:
            from farfinder.head.ui import run_head
            return run_head(args)
        return run(args)
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError, ValueError, cv2.error) as exc:
        LOG.error("%s", exc)
        return 1
