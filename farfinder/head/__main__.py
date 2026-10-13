"""Docker CLI for board generation, capture, reconstruction, and validation."""

import argparse
import json
import logging
import os
from pathlib import Path

import cv2

from farfinder.head.intrinsics import BoardSpec, IntrinsicCollector, Intrinsics, generate_board
from farfinder.head.session import HeadSession
from farfinder.head.capture_mode import CAPTURE_MODES


def camera_options(parser):
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--intrinsics", type=Path)
    parser.add_argument("--board-square-cm", type=float, default=2.0)


def scale_options(parser):
    parser.add_argument("--reference-square-cm", type=float,
                        help="Independently measured printed board square size; static board must appear in capture")
    parser.add_argument("--reference-uncertainty-cm", type=float,
                        help="Measurement uncertainty of that physical square dimension")
    parser.add_argument("--dense-backend", choices=["cpu","cuda"], default="cpu")


def ear_options(parser):
    parser.add_argument("--ear-model", type=Path)
    parser.add_argument("--ear-manifest", type=Path)


def calibrate_live(args):
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("Desktop DISPLAY required for intrinsic calibration")
    collector = IntrinsicCollector(BoardSpec(square_cm=args.board_square_cm))
    camera = cv2.VideoCapture(args.camera,cv2.CAP_V4L2)
    window = "ChArUco intrinsic calibration - automatic corners; Q cancels"
    try:
        if not camera.isOpened():
            raise RuntimeError("Cannot open webcam for intrinsic calibration")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH,args.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT,args.height)
        camera.set(cv2.CAP_PROP_BUFFERSIZE,1)
        cv2.namedWindow(window,cv2.WINDOW_NORMAL)
        while len(collector.views)<collector.required_views:
            ok,frame=camera.read()
            if not ok or frame is None:
                raise RuntimeError("Webcam stopped delivering frames")
            accepted,message=collector.add(frame)
            board=collector.detector.detect(frame)
            if board:
                import numpy as np
                for pixel in board["pixels"]:
                    cv2.circle(frame,tuple(np.rint(pixel).astype(int)),3,(0,220,255),-1)
            cv2.putText(frame,message,(15,30),cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,220,255),1,cv2.LINE_AA)
            cv2.imshow(window,frame)
            if cv2.waitKey(20)&0xFF in (ord("q"),27) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:
                raise RuntimeError("Intrinsic calibration cancelled; previous calibration preserved")
        intrinsic=collector.calibrate(args.camera)
        destination=args.intrinsics or args.data_dir/"intrinsics.json"
        intrinsic.save(destination)
        print(f"Saved {destination}; reprojection RMS {intrinsic.rms_px:.4f} px")
    finally:
        camera.release()
        cv2.destroyAllWindows()


def view_session(session):
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("Desktop DISPLAY required for 3D preview")
    from farfinder.head.pipeline import load_geometry
    from farfinder.head.viewer import MeshPreview
    mesh,surface,cameras,report=load_geometry(session)
    landmarks={}
    if (session.path/"landmarks.json").exists():
        landmarks=json.loads((session.path/"landmarks.json").read_text())["landmarks"]
    preview=MeshPreview()
    preview.set_mesh(mesh,next(iter(cameras.values()),None))
    window="Reconstructed head - CPU viewer (arrows rotate; Q quits)"
    cv2.namedWindow(window,cv2.WINDOW_NORMAL)
    try:
        while True:
            image=preview.render(landmarks,width=900,height=700)
            unit="cm reference validated" if report["metric_scale"]["valid"] else "Arbitrary reconstruction units"
            cv2.putText(image,unit,(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(100,210,255),1,cv2.LINE_AA)
            cv2.imshow(window,image)
            key=cv2.waitKeyEx(30)
            if key in (ord("q"),27) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:
                break
            if key in (81,65361,2424832): preview.yaw-=6
            if key in (83,65363,2555904): preview.yaw+=6
            if key in (82,65362,2490368): preview.pitch-=6
            if key in (84,65364,2621440): preview.pitch+=6
    finally:
        cv2.destroyWindow(window)


def main():
    parser=argparse.ArgumentParser(description="Local multi-view head reconstruction; anatomy/scale require validation")
    parser.add_argument("--data-dir",type=Path,default=Path("/app/data"))
    commands=parser.add_subparsers(dest="command",required=True)
    board=commands.add_parser("board",help="Generate printable ChArUco PNG/PDF")
    board.add_argument("--square-cm",type=float,default=2.0)
    calibrate=commands.add_parser("calibrate",help="Automatic one-time intrinsic calibration")
    camera_options(calibrate)
    live=commands.add_parser("live",help="Live webcam and 3D measurement interface")
    live.add_argument("--capture-mode",choices=CAPTURE_MODES,default="moving-camera")
    camera_options(live)
    scale_options(live)
    ear_options(live)
    live.add_argument("--session",type=Path)
    live.add_argument("--model",type=Path,default=Path("/app/models/face_landmarker.task"))
    reconstruction=commands.add_parser("reconstruct",help="COLMAP + CPU stereo + Open3D observed mesh")
    reconstruction.add_argument("session",type=Path)
    reconstruction.add_argument("--intrinsics",type=Path)
    scale_options(reconstruction)
    detection=commands.add_parser("detect",help="Local anatomy detection and camera/surface registration")
    detection.add_argument("session",type=Path)
    ear_options(detection)
    measurement=commands.add_parser("measure",help="Recalculate both distance modes")
    measurement.add_argument("session",type=Path)
    viewer=commands.add_parser("view",help="View mesh without a webcam")
    viewer.add_argument("session",type=Path)
    demo=commands.add_parser("demo",help="Real COLMAP reconstruction of synthetic multi-view head geometry")
    demo.add_argument("--orbit-views",type=int,default=24)
    demo.add_argument("--capture-mode",choices=CAPTURE_MODES,default="moving-camera")
    args=parser.parse_args()
    if getattr(args,"ear_model",None) and not getattr(args,"ear_manifest",None):
        parser.error("--ear-model requires --ear-manifest with anatomical labels and model checksum")
    if getattr(args,"ear_manifest",None) and not getattr(args,"ear_model",None):
        parser.error("--ear-manifest requires --ear-model")
    if (getattr(args,"reference_square_cm",None) is None) != (getattr(args,"reference_uncertainty_cm",None) is None):
        parser.error("Physical square dimension and its measurement uncertainty must be supplied together")
    if any(getattr(args,key,1)<=0 for key in ("width","height","fps")):
        parser.error("Camera dimensions and FPS must be positive")
    logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")
    try:
        if args.command=="board":
            generate_board(args.data_dir/"charuco_board",BoardSpec(square_cm=args.square_cm))
            print(f"Board saved to {args.data_dir/'charuco_board'}; print PDF at 100% and measure squares")
        elif args.command=="calibrate": calibrate_live(args)
        elif args.command=="live":
            from farfinder.head.ui import run_head
            return run_head(args)
        elif args.command=="demo":
            from farfinder.head.synthetic import generate_session,validate_reconstruction
            from farfinder.head.pipeline import reconstruct
            session=generate_session(args.data_dir,args.orbit_views,args.capture_mode)
            print(f"Synthetic session: {session.path}",flush=True)
            reconstruct(session)
            print(json.dumps(validate_reconstruction(session),indent=2))
        else:
            session=HeadSession.load(args.session)
            if args.command=="reconstruct":
                from farfinder.head.pipeline import reconstruct
                if args.intrinsics:
                    intrinsic=Intrinsics.load(args.intrinsics)
                    for frame in session.metadata["frames"]:
                        intrinsic.check_resolution(frame["width"],frame["height"])
                    session.metadata["intrinsics"]=intrinsic.to_dict()
                    session.save()
                reconstruct(session,args.reference_square_cm,args.reference_uncertainty_cm,args.dense_backend)
            elif args.command=="detect":
                from farfinder.head.ears import EarDetector,YoloOnnxEarDetector
                from farfinder.head.landmarks import detect_session
                ears=YoloOnnxEarDetector(args.ear_model,args.ear_manifest) if args.ear_model else EarDetector()
                detect_session(session,ears)
            elif args.command=="measure":
                from farfinder.head.pipeline import load_geometry
                from farfinder.head.measurements import calculate_measurements,save_measurements
                from farfinder.head.scale import MetricScale
                mesh,surface,cameras,report=load_geometry(session)
                payload=json.loads((session.path/"landmarks.json").read_text())
                results=calculate_measurements(payload["landmarks"],surface,MetricScale.from_dict(report["metric_scale"]),
                    payload.get("anatomical_frame"),report.get("coverage",{}).get("complete",False))
                save_measurements(session,results)
            elif args.command=="view": view_session(session)
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError,ValueError,RuntimeError,cv2.error) as error:
        logging.error("%s",error)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
