from dataclasses import asdict

import cv2
import numpy as np
import pytest

from farfinder.head.intrinsics import BoardSpec, BoardDetector, Intrinsics, IntrinsicCollector, generate_board
from farfinder.head.session import utc_now


def intrinsic():
    return Intrinsics(640,480,[[700,0,320],[0,710,240],[0,0,1]],[0.02,-0.01,0,0,0],0.3,[0.3]*20,
                      asdict(BoardSpec()),utc_now())


def test_intrinsic_roundtrip_and_resolution_gate(tmp_path):
    model=intrinsic()
    model.save(tmp_path/"intrinsics.json")
    loaded=Intrinsics.load(tmp_path/"intrinsics.json")
    assert np.allclose(loaded.K,model.K)
    assert np.allclose(loaded.D,model.D)
    loaded.check_resolution(640,480,"/dev/video0")
    with pytest.raises(ValueError,match="mismatch"):
        loaded.check_resolution(1280,720)
    with pytest.raises(ValueError,match="mismatch"):
        loaded.check_resolution(640,480,"/dev/video1")


@pytest.mark.parametrize("field,value",[("width",0),("camera_matrix",[[0,0,0]]*3),
    ("distortion",[0]*4),("rms_px",4),("rms_px",float("nan")),("schema_version",2)])
def test_invalid_intrinsics_rejected(field,value):
    payload=intrinsic().to_dict()|{field:value}
    with pytest.raises(ValueError): Intrinsics(**payload)


def test_printable_board_and_automatic_corner_detection(tmp_path):
    spec=BoardSpec()
    generate_board(tmp_path,spec)
    image=cv2.imread(str(tmp_path/"charuco.png"))
    image=cv2.copyMakeBorder(image,30,30,30,30,cv2.BORDER_CONSTANT,value=(255,255,255))
    detected=BoardDetector(spec).detect(image)
    assert len(detected["ids"]) >= 20
    assert (tmp_path/"charuco.pdf").is_file()
    assert (tmp_path/"board.json").is_file()


def test_intrinsic_calibration_recovers_known_synthetic_camera():
    collector=IntrinsicCollector(required_views=20)
    obj=collector.spec.board().getChessboardCorners()
    k=np.array([[700.,0,320],[0,710,240],[0,0,1]])
    distortion=np.array([0.01,-0.015,0.001,-0.001,0])
    rng=np.random.default_rng(11)
    collector.resolution=(640,480)
    for i in range(24):
        rotation=rng.uniform(-0.45,0.45,3)
        translation=np.array([-6+rng.uniform(-5,5),-4+rng.uniform(-3,3),rng.uniform(32,65)])
        pixels,_=cv2.projectPoints(obj,rotation,translation,k,distortion)
        collector.views.append(dict(ids=list(range(len(obj))),pixels=pixels.reshape(-1,2).tolist()))
        # Deliberately varied coverage; exact camera projections are the ground truth.
        collector.signatures.append(np.array([i%4/4,i%6/6,0,0,0,0]))
    recovered=collector.calibrate()
    assert recovered.K[0,0]==pytest.approx(700,rel=0.002)
    assert recovered.K[1,1]==pytest.approx(710,rel=0.002)
    assert np.allclose(recovered.D,distortion,atol=0.005)
    assert recovered.rms_px<0.01


def test_insufficient_board_views_cannot_calibrate():
    with pytest.raises(ValueError,match="Need"):
        IntrinsicCollector().calibrate()
