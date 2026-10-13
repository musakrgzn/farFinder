"""Webcam/3D head interface, automatic capture, and optional G/O-only image clicks."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dataclasses import asdict
import json
import logging
import os
import queue
import signal
import threading
import time

import cv2
import numpy as np

from farfinder.head.capture import CaptureSelector, HeadSegmenter, VIEW_NAMES
from farfinder.head.capture_mode import capture_assumptions, head_turn_coverage
from farfinder.head.ears import EarDetector, YoloOnnxEarDetector
from farfinder.head.intrinsics import BoardSpec, BoardDetector, IntrinsicCollector, Intrinsics
from farfinder.head.landmarks import associate_selections, detect_session, unresolved_landmarks
from farfinder.head.measurements import calculate_measurements, save_measurements
from farfinder.head.pipeline import reconstruct, load_geometry
from farfinder.head.scale import MetricScale
from farfinder.head.session import HeadSession, save_json
from farfinder.head.viewer import MeshPreview
from farfinder.measurement import MeasurementEngine
from farfinder.storage import DataStore, MeasurementLogger
from farfinder.calibration import CalibrationError
from farfinder.vision import FaceDetector, draw_landmarks


WINDOW = "farFinder - automatic head capture and 3D measurement"
BUTTONS = [("Start capture [S]", "s"), ("Stop capture [T]", "t"), ("Reconstruct [R]", "r"),
           ("Detect landmarks [D]", "d"), ("Measurements [M]", "m"), ("Euclidean [E]", "e"),
           ("Surface paths [U]", "u"), ("Save results [W]", "w"), ("Intrinsics [C]", "c"),
           ("Optional G/O [B]", "b")]


def _text(image, text, x, y, color=(230,230,230), size=0.47):
    cv2.putText(image, str(text), (x,y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def browse_selections(session, landmarks, cameras, surface):
    window = "Captured images - optional G/O only (not Kang V/P)"
    frames = session.metadata["frames"]
    if not frames:
        raise ValueError("No captured images")
    index, selected = 0, "G"
    state = dict(message="G=glabella; O=posterior skull/base. Clicking is optional.")
    def click(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        try:
            session.select(selected, frames[index]["image"], [x,y])
            associate_selections(session, cameras, surface, landmarks)
            state["message"] = f"{selected}: " + landmarks[selected]["reason"]
        except ValueError as error:
            state["message"] = str(error)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, click)
    try:
        while True:
            frame = frames[index]
            image = cv2.imread(str(session.path / "images" / frame["image"]))
            if image is None:
                raise ValueError("Captured image missing")
            height, width = image.shape[:2]
            canvas = np.full((height+120, max(width, 900), 3), 25, np.uint8)
            canvas[:height,:width] = image  # Original, unmirrored, unscaled image coordinates.
            for item in session.metadata["selections"]:
                if item["image"] == frame["image"]:
                    point = tuple(np.rint(item["pixel"]).astype(int))
                    cv2.circle(canvas, point, 5, (0,200,255), 2)
                    _text(canvas, item["name"], point[0]+8, point[1]-5)
            _text(canvas, f"{index+1}/{len(frames)} {frame['image']} | Active selection: {selected}", 15, height+25)
            _text(canvas, "G/O: choose permitted reference | N/P: next/previous image | Q/Esc: return", 15, height+52)
            _text(canvas, state["message"][:105], 15, height+83, (100,190,255), 0.43)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("g"), ord("o")):
                selected = chr(key).upper()
            elif key == ord("n"):
                index = (index+1) % len(frames)
            elif key == ord("p"):
                index = (index-1) % len(frames)
    finally:
        cv2.destroyWindow(window)
    path = session.path / "landmarks.json"
    payload = json.loads(path.read_text()) if path.exists() else dict(schema_version=1, observations=[])
    payload["landmarks"] = landmarks
    save_json(path, payload)


def run_head(args):
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("Ubuntu X11/XWayland DISPLAY is required for the head interface")
    data_dir = Path(args.data_dir)
    capture_mode = getattr(args, "capture_mode", "moving-camera")
    assumptions = capture_assumptions(capture_mode)
    store = DataStore(data_dir)
    intrinsic_path = Path(getattr(args, "intrinsics", None) or data_dir / "intrinsics.json")
    intrinsic = Intrinsics.load(intrinsic_path) if intrinsic_path.exists() else None
    spec = BoardSpec(**intrinsic.board) if intrinsic else BoardSpec(square_cm=getattr(args,"board_square_cm",2.0))
    board_detector = BoardDetector(spec, intrinsic)
    ear_detector = (YoloOnnxEarDetector(args.ear_model, args.ear_manifest)
                    if getattr(args,"ear_model",None) and getattr(args,"ear_manifest",None) else EarDetector())
    camera = None
    selector, preview = CaptureSelector(), MeshPreview()
    session = HeadSession.load(args.session) if getattr(args,"session",None) else None
    if session:
        capture_mode = session.metadata.get("capture_mode", "moving-camera")
        assumptions = capture_assumptions(capture_mode)
    landmarks, measurements = unresolved_landmarks(), []
    surface, cameras = None, {}
    if session:
        try:
            mesh, surface, cameras, report = load_geometry(session)
            preview.set_mesh(mesh, next(iter(cameras.values()), None))
            if (session.path / "landmarks.json").exists():
                landmarks = json.loads((session.path / "landmarks.json").read_text())["landmarks"]
            if (session.path / "measurements.json").exists():
                measurements = json.loads((session.path / "measurements.json").read_text())["measurements"]
        except RuntimeError:
            pass
    messages = queue.SimpleQueue()
    pending_keys = queue.SimpleQueue()
    cancel = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    future = None
    capturing, collector = False, None
    message = "Intrinsics [C], then capture [S]. " + assumptions["instructions"]
    mode = "euclidean"
    segmenter = detector = logger = None
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        cancel.set()
    previous = signal.signal(signal.SIGTERM, stop)
    def click(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN or not 575 <= y < 663:
            return
        row = 0 if y < 619 else 1
        column = min(4, x//256)
        pending_keys.put(ord(BUTTONS[row*5+column][1]))
    try:
        camera = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
        if not camera.isOpened():
            raise RuntimeError(f"Cannot open {args.camera}; check webcam mapping and VIDEO_GID")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        camera.set(cv2.CAP_PROP_FPS, args.fps)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = camera.read()
        if not ok or frame is None:
            raise RuntimeError("Webcam delivered no image")
        height, width = frame.shape[:2]
        if intrinsic:
            try:
                intrinsic.check_resolution(width, height, args.camera)
            except ValueError as error:
                message = str(error)
                intrinsic = None
                board_detector = BoardDetector(spec)
        try:
            distance_calibration = store.load(args.camera, width, height)
        except CalibrationError:
            distance_calibration = None
        engine = MeasurementEngine(distance_calibration)
        detector = FaceDetector(Path(args.model))
        segmenter = HeadSegmenter()
        logger = MeasurementLogger(data_dir)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1280, 850)
        cv2.setMouseCallback(WINDOW, click)
        started, last = time.monotonic(), time.monotonic()
        fps = 0
        while not stopping:
            now = time.monotonic()
            fps = 0.2/max(now-last, 1e-6)+0.8*fps
            last = now
            current_height, current_width = frame.shape[:2]
            if (current_width,current_height) != (width,height):
                capturing = False
                engine.set_calibration(None)
                intrinsic = None
                raise RuntimeError("Camera resolution changed; restart and recalibrate")
            observation = detector.detect(frame, int((now-started)*1000))
            reading = engine.process(observation.span_px, observation.quality)
            logger.write(now, reading, observation.span_px, fps, engine.calibration, observation.pose)
            shown = draw_landmarks(frame, observation)
            # Do not call the same OpenCV DNN instance concurrently with the
            # background session detector; unknown anatomy has no preview point.
            if future is None or future.done():
                for name,item in ear_detector.detect(frame).items():
                    if item.get("visible") and item.get("pixel") is not None:
                        point=tuple(np.rint(item["pixel"]).astype(int))
                        color=(80,220,110) if item.get("anatomical_validated") else (60,190,255)
                        cv2.circle(shown,point,4,color,-1,cv2.LINE_AA)
                        _text(shown,name,point[0]+6,point[1]-5,color,0.35)
            if capturing:
                head = segmenter.detect(frame)
                accepted, quality = selector.evaluate(frame, head, now, observation.pose)
                message = quality["reason"]
                tint = [30,80,30] if head.source!="temporal_tracking" else [30,150,220]
                shown[head.mask > 0] = (shown[head.mask > 0]*0.8+np.array(tint)*0.2).astype(np.uint8)
                if accepted:
                    pose = None if not observation.pose else dict(yaw_deg=observation.pose[0], pitch_deg=observation.pose[1],
                                                                 roll_deg=observation.pose[2], status="ESTIMATED")
                    session.add_frame(frame, head.mask, head.categories, quality, pose, board_detector.detect(frame))
                    message = f"Saved {len(session.metadata['frames'])} overlapping head frames"
                    if head.source=="temporal_tracking":
                        message += " - head region tracked while facial cues are unavailable"
                    if len(session.metadata["frames"]) >= 100:
                        capturing = False
                        session.stop()
                        message = "100-frame session limit reached; run reconstruction"
            if collector:
                accepted, message = collector.add(frame)
                board = collector.detector.detect(frame)
                if board:
                    for pixel in board["pixels"]:
                        cv2.circle(shown, tuple(np.rint(pixel).astype(int)), 3, (0,255,255), -1)
                if len(collector.views) >= collector.required_views:
                    try:
                        intrinsic = collector.calibrate(args.camera)
                        intrinsic.save(intrinsic_path)
                        board_detector = BoardDetector(spec, intrinsic)
                        collector = None
                        message = f"Intrinsic calibration saved; RMS {intrinsic.rms_px:.3f} px"
                    except ValueError as error:
                        collector = None
                        message = str(error)+". Press C to restart."
            while not messages.empty():
                message = messages.get()
            if future and future.done():
                try:
                    future.result()
                    try:
                        mesh, surface, cameras, report = load_geometry(session)
                        preview.set_mesh(mesh, next(iter(cameras.values()), None))
                    except RuntimeError:
                        pass
                    if (session.path / "landmarks.json").exists():
                        landmarks = json.loads((session.path / "landmarks.json").read_text())["landmarks"]
                    if (session.path / "measurements.json").exists():
                        measurements = json.loads((session.path / "measurements.json").read_text())["measurements"]
                except Exception as error:
                    message = f"FAILED: {error}"
                    logging.error("Head processing failed: %s", error)
                future = None
            canvas = np.full((860,1280,3), 24, np.uint8)
            canvas[:480,:640] = cv2.resize(shown,(640,480))
            canvas[:480,640:] = preview.render(landmarks)
            live_title = "LIVE WEBCAM - head turns (experimental)" if capture_mode=="head-rotation" else "LIVE WEBCAM - facial references"
            _text(canvas, live_title, 12, 26, (100,230,150),0.57)
            _text(canvas, "OBSERVED 3D HEAD - arrows rotate view", 650,26,(255,210,100),0.57)
            distance = "-- cm" if reading.distance_cm is None else f"{reading.distance_cm:.1f} cm"
            _text(canvas, f"Existing distance: {distance} | FPS {fps:.1f} | {reading.quality}", 12,505,size=0.47)
            scale = MetricScale.from_dict(session.metadata.get("metric_scale",{})) if session else MetricScale()
            _text(canvas, "Metric scale: " + ("VALIDATED cm" if scale.valid else "UNRESOLVED - arbitrary reconstruction units"),650,505,(100,230,150) if scale.valid else (80,180,255))
            _text(canvas, message[:145],12,538,(120,200,255),0.48)
            count = len(session.metadata["frames"]) if session else 0
            capture_label = "Head turns (experimental)" if capture_mode=="head-rotation" else "Moving camera"
            yaw_text = ""
            if capture_mode=="head-rotation" and session:
                coverage = head_turn_coverage(session.metadata["frames"])
                yaw_text = f" | Saved yaw span {coverage['yaw_span_deg']:.0f} deg"
            _text(canvas, f"Frames {count} | {'CAPTURING' if capturing else 'STOPPED'} | Intrinsics {'VALID' if intrinsic else 'MISSING'} | {capture_label} | {mode}{yaw_text}",12,563,size=0.42)
            for index,(title,key) in enumerate(BUTTONS):
                row,column = divmod(index,5)
                x,y = column*256,575+row*44
                cv2.rectangle(canvas,(x+3,y+3),(x+252,y+39),(65,57,50),-1)
                _text(canvas,title,x+12,y+27,size=0.51)
            _text(canvas, "Landmarks (green DETECTED / amber ESTIMATED / red UNRESOLVED)",12,687,size=0.44)
            for index,name in enumerate(["G","O","left_tragus","right_tragus","left_superior_ear_root","right_superior_ear_root","V","P","FL","FR"]):
                item = landmarks.get(name,{"status":"UNRESOLVED"})
                status = item["status"]
                color = (80,220,110) if status=="DETECTED" else (60,180,255) if status=="ESTIMATED" else (90,100,220)
                x,y = 12+(index//5)*300,710+(index%5)*22
                _text(canvas,f"{name}: {status}",x,y,color,0.43)
            selected_type = "euclidean" if mode=="euclidean" else "surface_mesh_edge_approximation"
            displayed = [item for item in measurements if item["distance_type"] in (selected_type,"anatomical_axis_extent")]
            for index,item in enumerate(displayed[:9]):
                value = "UNAVAILABLE" if item["distance_value"] is None else f"{item['distance_value']:.3f} {item['distance_unit']}"
                uncertainty = (item.get("confidence_or_uncertainty") or {}).get("geometric_uncertainty")
                if uncertainty is not None:
                    value += f" +/- {uncertainty:.3f} (proxy)"
                _text(canvas, item["measurement_name"]+": "+value,650,689+index*17,size=0.40)
            _text(canvas,"Q/Esc: quit | G/O optional clicks are not Kang V/P | No anatomical accuracy claim",12,846,(160,160,160),0.45)
            cv2.imshow(WINDOW,canvas)
            key = cv2.waitKeyEx(1)
            if not pending_keys.empty():
                key = pending_keys.get()
            if key in (ord("q"),27) or cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1:
                break
            if key in (81,65361,2424832): preview.yaw -= 6
            if key in (83,65363,2555904): preview.yaw += 6
            if key in (82,65362,2490368): preview.pitch -= 6
            if key in (84,65364,2621440): preview.pitch += 6
            if future and key in [ord(value) for value in "srdmbcw"]:
                message = "Processing is running; wait until completion"
            elif key == ord("s"):
                collector = None
                session = HeadSession.create(data_dir,intrinsic,engine.calibration,capture_mode=capture_mode)
                selector = CaptureSelector()
                selector.capture_mode = capture_mode
                segmenter.reset_tracking()
                landmarks,measurements = unresolved_landmarks(),[]
                surface,cameras = None,{}
                preview = MeshPreview()
                capturing = True
                message = assumptions["instructions"]
            elif key == ord("t"):
                if session and capturing: session.stop()
                capturing = False
                message = "Capture stopped. Reconstruction requires intrinsics and sufficient overlap."
            elif key == ord("c"):
                if capturing:
                    message = "Stop head capture before intrinsic calibration"
                else:
                    collector = IntrinsicCollector(spec)
                    message = "Show printed ChArUco board; vary tilt, size, and position automatically."
            elif key in (ord("r"),ord("d")):
                if capturing or not session or not session.metadata["frames"]:
                    message = "Stop a captured session before processing"
                elif collector:
                    message = "Finish or restart intrinsic calibration before processing"
                else:
                    cancel.clear()
                    if key==ord("r"):
                        landmarks,measurements=unresolved_landmarks(),[]
                        surface,cameras=None,{}
                        preview=MeshPreview()
                        if not session.metadata.get("intrinsics") and intrinsic:
                            session.metadata["intrinsics"] = intrinsic.to_dict()
                            session.save()
                        future = executor.submit(reconstruct,session,getattr(args,"reference_square_cm",None),
                            getattr(args,"reference_uncertainty_cm",None),getattr(args,"dense_backend","cpu"),cancel,messages.put)
                    else:
                        future = executor.submit(detect_session,session,ear_detector,None,messages.put)
            elif key==ord("m"):
                try:
                    mesh,surface,cameras,report = load_geometry(session)
                    payload=json.loads((session.path/"landmarks.json").read_text())
                    measurements=calculate_measurements(payload["landmarks"],surface,MetricScale.from_dict(report["metric_scale"]),
                                                       payload.get("anatomical_frame"),report.get("coverage",{}).get("complete",False))
                    save_measurements(session,measurements)
                    message="Measurements saved; unresolved references remain unavailable"
                except (OSError,ValueError,RuntimeError,AttributeError) as error:
                    message=str(error)
            elif key==ord("e"): mode="euclidean"
            elif key==ord("u"): mode="surface"
            elif key==ord("w"):
                if session:
                    session.save()
                    message=f"Results saved: {session.path}"
            elif key==ord("b"):
                if not session or capturing:
                    message="Stop capture to browse optional G/O selections; capture does not require clicks"
                else:
                    try:
                        browse_selections(session,landmarks,cameras,surface)
                        message="Optional selections saved. Press M to recalculate measurements."
                    except (OSError,ValueError) as error:
                        message=str(error)
            if not stopping:
                ok,frame=camera.read()
                if not ok or frame is None:
                    raise RuntimeError("Webcam stopped delivering frames")
    finally:
        cancel.set()
        executor.shutdown(wait=True,cancel_futures=True)
        if session and capturing: session.stop()
        for resource in (logger,segmenter,detector):
            if resource: resource.close()
        if camera: camera.release()
        cv2.destroyAllWindows()
        signal.signal(signal.SIGTERM,previous)
    return 0
