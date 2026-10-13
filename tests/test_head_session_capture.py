import argparse
from dataclasses import asdict
import json
from pathlib import Path
import signal
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from farfinder.calibration import Calibration
from farfinder.head.capture import HeadMask,HeadSegmenter,CaptureSelector
from farfinder.head.colmap import ColmapRunner,ReconstructionError,read_text_model,head_support
from farfinder.head.geometry import CameraPose
from farfinder.head.intrinsics import BoardSpec,Intrinsics
from farfinder.head.pipeline import reconstruct
from farfinder.head.session import HeadSession,utc_now
from farfinder.head.synthetic import look_at,texture_map,render_ellipsoid
from farfinder.head.ui import run_head
from farfinder.head.viewer import MeshPreview
from farfinder.storage import DataStore
from farfinder.calibration import CaptureResult
from farfinder.vision import Observation


def test_session_serialization_retains_calibrations_images_and_observations(tmp_path):
    k=[[700,0,320],[0,700,240],[0,0,1]]
    intrinsic=Intrinsics(640,480,k,[0]*5,0,[0]*20,asdict(BoardSpec()),utc_now())
    distance=Calibration.fit(200,100,"/dev/video0",640,480)
    session=HeadSession.create(tmp_path,intrinsic,distance)
    image=np.zeros((480,640,3),np.uint8)
    name=session.add_frame(image,np.full((480,640),255,np.uint8),np.zeros((480,640),np.uint8),
                           dict(status="ACCEPTED",sharpness=200),dict(yaw_deg=0,status="ESTIMATED"))
    session.select("O",name,[20,30])
    session.stop()
    loaded=HeadSession.load(session.path)
    assert loaded.metadata["distance_calibration"]==distance.to_dict()
    assert loaded.metadata["intrinsics"]==intrinsic.to_dict()
    assert loaded.metadata["frames"][0]["camera_pose"] is None
    assert loaded.metadata["selections"][0]["point_3d"] is None
    assert (session.path/"images"/name).is_file()
    assert (session.path/"masks"/(name+".png")).is_file()
    with pytest.raises(ValueError,match="resolution"):
        session.add_frame(np.zeros((100,100,3),np.uint8),np.zeros((100,100),np.uint8),np.zeros((100,100),np.uint8),{})


def test_session_rejects_traversal_and_unpermitted_manual_points(tmp_path):
    session=HeadSession.create(tmp_path)
    payload=session.metadata|dict(frames=[dict(image="../outside.png",width=10,height=10)])
    (session.path/"session.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError): HeadSession.load(session.path)
    session.save()
    payload=session.metadata|dict(selections=[dict(name="FL",pixel=[1,1])])
    (session.path/"session.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError,match="Only optional G and O"):
        HeadSession.load(session.path)


def test_automatic_selector_accepts_texture_and_rejects_blur_and_duplicates():
    k=np.array([[700.,0,320],[0,700,240],[0,0,1]])
    rotation,translation=look_at([0,0,48])
    camera=CameraPose("synthetic",k,rotation,translation,np.zeros(5),640,480)
    image,mask=render_ellipsoid(camera,[8,12,10],texture_map())
    head=HeadMask(mask,(mask>0).astype(np.uint8),np.ones(mask.shape))
    selector=CaptureSelector()
    accepted,quality=selector.evaluate(image,head,0)
    assert accepted
    assert quality["feature_count"]>=70
    assert not selector.evaluate(image,head,1)[0]
    blurred=cv2.GaussianBlur(image,(0,0),8)
    assert not CaptureSelector().evaluate(blurred,head,2)[0]
    missing=HeadMask(np.zeros(mask.shape,np.uint8),head.categories,head.confidence,"Head unresolved")
    assert not CaptureSelector().evaluate(image,missing,3)[0]


def test_native_segmentation_model_runs_without_camera_or_network():
    segmenter=HeadSegmenter()
    try:
        assert segmenter.segmenter.labels==["background","hair","body-skin","face-skin","clothes","others"]
        mask=segmenter.detect(np.zeros((240,320,3),np.uint8))
        assert not np.any(mask.mask)
        assert "No reliable" in mask.reason
    finally:
        segmenter.close()


def test_failed_reconstruction_saves_clear_error_without_fake_mesh(tmp_path):
    session=HeadSession.create(tmp_path)
    with pytest.raises(ReconstructionError,match="Intrinsic calibration"):
        reconstruct(session)
    report=json.loads((session.path/"quality_report.json").read_text())
    assert report["reconstruction_status"]=="FAILED"
    assert not report["metric_scale"]["valid"]
    assert not (session.path/"head_mesh.ply").exists()
    assert all(item["distance_value"] is None for item in json.loads((session.path/"measurements.json").read_text())["measurements"])


def test_reconstruction_restart_invalidates_old_optional_3d_association(tmp_path):
    session=HeadSession.create(tmp_path)
    image=np.zeros((100,100,3),np.uint8)
    session.add_frame(image,np.full((100,100),255,np.uint8),np.zeros((100,100),np.uint8),{})
    selection=session.select("G","frame_00000.png",[50,50])
    selection.update(status="DETECTED",point_3d=[1,2,3])
    session.save()
    with pytest.raises(ReconstructionError): reconstruct(session)
    restored=HeadSession.load(session.path)
    assert restored.metadata["selections"][0]["point_3d"] is None
    assert restored.metadata["selections"][0]["status"]=="OBSERVATION_2D_ONLY"


def test_colmap_text_parser_preserves_pose_and_all_calibrated_distortion(tmp_path):
    (tmp_path/"cameras.txt").write_text("1 FULL_OPENCV 640 480 500 510 320 240 .01 .02 .001 .002 .003 0 0 0\n")
    text=""
    for index,name in enumerate(("a.png","b.png","c.png"),1):
        text+=f"{index} 1 0 0 0 {index-2} 0 0 1 {name}\n320 240 9\n"
    (tmp_path/"images.txt").write_text(text)
    (tmp_path/"points3D.txt").write_text("9 0 0 5 255 128 0 .2 1 0 2 0 3 0\n")
    poses,points,colors,errors,tracks=read_text_model(tmp_path)
    assert len(poses)==3
    assert poses["a.png"].distortion==pytest.approx([.01,.02,.001,.002,.003])
    assert points[0]==pytest.approx([0,0,5])
    assert len(tracks[0])==3


def test_background_points_are_rejected_by_head_masks():
    k=np.array([[500.,0,320],[0,500,240],[0,0,1]])
    cameras={name:CameraPose(name,k,np.eye(3),[-center,0,0],np.zeros(5),640,480)
             for name,center in (("a",-1),("b",0),("c",1))}
    masks={name:np.zeros((480,640),np.uint8) for name in cameras}
    for mask in masks.values(): mask[180:300,150:490]=255
    keep,support=head_support([[0,0,5],[4,0,5]],cameras,masks)
    assert keep.tolist()==[True,False]


def test_head_ui_camera_failure_releases_resources_and_signal(tmp_path,monkeypatch):
    monkeypatch.setenv("DISPLAY",":0")
    camera=Mock()
    camera.isOpened.return_value=False
    monkeypatch.setattr("farfinder.head.ui.cv2.VideoCapture",lambda *args:camera)
    monkeypatch.setattr("farfinder.head.ui.cv2.destroyAllWindows",Mock())
    previous=signal.getsignal(signal.SIGTERM)
    args=argparse.Namespace(data_dir=tmp_path,camera="/dev/video0")
    with pytest.raises(RuntimeError,match="Cannot open"):
        run_head(args)
    camera.release.assert_called_once()
    assert signal.getsignal(signal.SIGTERM)==previous


@pytest.mark.parametrize("capture_mode",["moving-camera","head-rotation"])
def test_head_ui_start_stop_preserves_distance_calibration(tmp_path,monkeypatch,capture_mode):
    monkeypatch.setenv("DISPLAY",":0")
    calibration=Calibration.fit(200,100,"/dev/video0",640,480)
    DataStore(tmp_path).save(calibration,[CaptureResult(30,200,45,0),CaptureResult(60,100,45,0)])
    camera=Mock()
    image=np.full((480,640,3),100,np.uint8)
    camera.isOpened.return_value=True
    camera.read.return_value=(True,image)
    monkeypatch.setattr("farfinder.head.ui.cv2.VideoCapture",lambda *args:camera)
    detector=Mock()
    detector.detect.return_value=Observation(200,"OK",pose=(0,0,0))
    monkeypatch.setattr("farfinder.head.ui.FaceDetector",lambda *args:detector)
    segmenter=Mock()
    mask=np.zeros((480,640),np.uint8)
    mask[80:400,150:490]=255
    segmenter.detect.return_value=HeadMask(mask,(mask>0).astype(np.uint8),np.ones(mask.shape))
    monkeypatch.setattr("farfinder.head.ui.HeadSegmenter",lambda:segmenter)
    selector=Mock()
    selector.evaluate.return_value=(True,dict(reason="Accepted",status="ACCEPTED"))
    monkeypatch.setattr("farfinder.head.ui.CaptureSelector",lambda:selector)
    keys=iter([ord("s"),ord("t"),ord("q")])
    monkeypatch.setattr("farfinder.head.ui.cv2.waitKeyEx",lambda delay:next(keys))
    monkeypatch.setattr("farfinder.head.ui.cv2.getWindowProperty",lambda *args:1)
    for method in ("namedWindow","resizeWindow","setMouseCallback","imshow","destroyAllWindows"):
        monkeypatch.setattr(f"farfinder.head.ui.cv2.{method}",Mock())
    args=argparse.Namespace(data_dir=tmp_path,camera="/dev/video0",width=640,height=480,fps=30,
                            model=Path("/app/models/face_landmarker.task"),capture_mode=capture_mode)
    assert run_head(args)==0
    session=HeadSession.load(next((tmp_path/"head_sessions").iterdir()))
    assert len(session.metadata["frames"])==1
    assert session.metadata["state"]=="CAPTURED"
    assert session.metadata["capture_mode"]==capture_mode
    assert session.metadata["distance_calibration"]==calibration.to_dict()
    assert DataStore(tmp_path).load("/dev/video0",640,480)==calibration
    detector.close.assert_called_once()
    segmenter.close.assert_called_once()
    camera.release.assert_called_once()


def test_mesh_preview_is_cpu_raster_image():
    import open3d as o3d
    viewer=MeshPreview()
    viewer.set_mesh(o3d.geometry.TriangleMesh.create_sphere(radius=1,resolution=8))
    first=viewer.render(width=400,height=300)
    assert first.shape==(300,400,3)
    assert np.unique(first.reshape(-1,3),axis=0).shape[0]>10
