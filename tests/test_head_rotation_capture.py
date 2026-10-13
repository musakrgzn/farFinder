"""Head-turn safeguards and known rigid-motion projection validation."""

from dataclasses import asdict
import json

import cv2
import numpy as np
import pytest

from farfinder.head.capture import CaptureSelector, HeadMask
from farfinder.head.capture_mode import capture_assumptions, validate_head_turns
from farfinder.head.geometry import CameraPose, camera_relative_to_object
from farfinder.head.intrinsics import BoardSpec, Intrinsics
from farfinder.head.pipeline import reconstruct
from farfinder.head.colmap import ReconstructionError
from farfinder.head.scale import scale_from_board
from farfinder.head.session import HeadSession, utc_now
from farfinder.head.synthetic import look_at, render_ellipsoid, texture_map


def frames(yaws):
    return [dict(image=f"frame_{index:05d}.png",width=640,height=480,
                 estimated_head_orientation=dict(yaw_deg=float(yaw),pitch_deg=0,roll_deg=0))
            for index,yaw in enumerate(yaws)]


def intrinsic():
    return Intrinsics(640,480,[[700,0,320],[0,700,240],[0,0,1]], [0]*5,0,[0]*20,
                      asdict(BoardSpec(square_cm=3.3)),utc_now())


def test_fixed_camera_moving_object_projects_in_head_relative_coordinates():
    camera=CameraPose("fixed",np.array([[500.,0,320],[0,500,240],[0,0,1]]),
                      np.eye(3),np.zeros(3),np.zeros(5),640,480)
    rotation,_=cv2.Rodrigues(np.array([0.1,0.5,-0.08]))
    translation=np.array([0.2,-0.1,5.])
    points=np.random.default_rng(2).uniform(-0.4,0.4,(30,3))
    relative=camera_relative_to_object(camera,rotation,translation)
    actual_pixels,_=camera.project(points @ rotation.T+translation)
    relative_pixels,_=relative.project(points)
    assert np.allclose(actual_pixels,relative_pixels,atol=1e-8)
    assert not np.allclose(relative.center,camera.center)
    assert camera.center==pytest.approx([0,0,0])


def test_object_motion_requires_a_rigid_rotation():
    camera=CameraPose("fixed",np.eye(3),np.eye(3),np.zeros(3),np.zeros(5),10,10)
    with pytest.raises(ValueError):
        camera_relative_to_object(camera,np.diag([1,2,1]),[0,0,5])


def test_fixed_background_preserves_object_pixels_outside_eroded_feature_mask():
    k=np.array([[700.,0,320],[0,700,240],[0,0,1]])
    rotation,translation=look_at([0,0,48])
    camera=CameraPose("",k,rotation,translation,np.zeros(5),640,480)
    texture=np.full((100,200,3),100,np.uint8)
    original,mask=render_ellipsoid(camera,[8,12,10],texture)
    background=np.full(original.shape,200,np.uint8)
    image,_=render_ellipsoid(camera,[8,12,10],texture,background=background)
    object_pixels=np.all(original==100,axis=2)
    edge=object_pixels&(mask==0)
    assert np.any(edge)
    assert np.all(image[object_pixels]==100)
    assert np.all(image[~object_pixels]==200)


def test_head_turn_session_roundtrip_and_legacy_default(tmp_path):
    session=HeadSession.create(tmp_path,intrinsic(),capture_mode="head-rotation")
    image=np.zeros((480,640,3),np.uint8)
    session.add_frame(image,np.zeros((480,640),np.uint8),np.zeros((480,640),np.uint8),{},
                      dict(yaw_deg=0,pitch_deg=0,roll_deg=0))
    loaded=HeadSession.load(session.path)
    assert loaded.metadata["capture_mode"]=="head-rotation"
    assert loaded.metadata["pose_coordinate_frame"]=="head_relative"
    assert loaded.metadata["intrinsics"]["board"]["square_cm"]==3.3
    payload=loaded.metadata.copy()
    del payload["capture_mode"]
    (session.path/"session.json").write_text(json.dumps(payload))
    legacy=HeadSession.load(session.path)
    assert capture_assumptions(legacy.metadata.get("capture_mode","moving-camera"))["capture_mode"]=="moving-camera"


def test_invalid_capture_mode_rejected_before_session_creation(tmp_path):
    with pytest.raises(ValueError,match="Unsupported capture mode"):
        HeadSession.create(tmp_path,capture_mode="unknown")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("angles",[[0]*12,list(range(12)),list(range(0,60,5))])
def test_narrow_or_one_sided_head_turns_rejected(angles):
    with pytest.raises(ValueError,match="Head-turn views too similar"):
        validate_head_turns(frames(angles))


def test_head_turns_need_enough_views_and_both_sides():
    with pytest.raises(ValueError,match="at least 12"):
        validate_head_turns(frames(np.linspace(-45,45,11)))
    coverage=validate_head_turns(frames(np.linspace(-45,45,19)))
    assert coverage["both_sides"] and coverage["frontal_anchor"]
    assert coverage["yaw_span_deg"]==pytest.approx(90)


def test_stationary_board_cannot_scale_rotating_head(tmp_path):
    session=HeadSession.create(tmp_path,intrinsic(),capture_mode="head-rotation")
    scale=scale_from_board(session,{},measured_square_cm=3.3,uncertainty_cm=0.02)
    assert not scale.valid and scale.cm_per_unit is None
    assert "stationary board" in scale.reason


def test_invalid_head_turn_capture_fails_before_colmap(tmp_path,monkeypatch):
    session=HeadSession.create(tmp_path,intrinsic(),capture_mode="head-rotation")
    session.metadata["frames"]=frames([0]*12)
    def unexpected(*args,**kwargs):
        pytest.fail("COLMAP must not run on inadequate head-turn views")
    monkeypatch.setattr("farfinder.head.pipeline.ColmapRunner.reconstruct_sparse",unexpected)
    with pytest.raises(ReconstructionError,match="Head-turn views too similar"):
        reconstruct(session)
    report=json.loads((session.path/"quality_report.json").read_text())
    assert report["reconstruction_status"]=="FAILED"
    assert report["experimental_capture"]
    assert not report["metric_scale"]["valid"]
    assert not (session.path/"head_mesh.ply").exists()


def test_head_turn_selector_starts_frontal_and_accepts_real_rigid_overlap():
    k=np.array([[700.,0,320],[0,700,240],[0,0,1]])
    rotation,translation=look_at([0,0,48])
    camera=CameraPose("",k,rotation,translation,np.zeros(5),640,480)
    texture=texture_map()
    image,mask=render_ellipsoid(camera,[8,12,10],texture)
    head=HeadMask(mask,(mask>0).astype(np.uint8),np.ones(mask.shape))
    selector=CaptureSelector(capture_mode="head-rotation")
    assert "Start head-turn" in selector.evaluate(image,head,0,pose=(30,0,0))[1]["reason"]
    assert selector.evaluate(image,head,0,pose=(0,0,0))[0]
    rotation,translation=look_at([48*np.sin(np.radians(7)),0,48*np.cos(np.radians(7))])
    relative=CameraPose("",k,rotation,translation,np.zeros(5),640,480)
    object_rotation=camera.R.T @ relative.R
    object_translation=camera.R.T @ (relative.t-camera.t)
    moving_view=camera_relative_to_object(camera,object_rotation,object_translation)
    image,mask=render_ellipsoid(moving_view,[8,12,10],texture)
    head=HeadMask(mask,(mask>0).astype(np.uint8),np.ones(mask.shape))
    accepted,quality=selector.evaluate(image,head,1,pose=(7,0,0))
    assert accepted,quality
    assert quality["capture_mode"]=="head-rotation"
    assert quality["geometric_inlier_fraction"]>=0.65
