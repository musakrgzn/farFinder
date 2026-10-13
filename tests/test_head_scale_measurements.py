from dataclasses import asdict

import numpy as np
import open3d as o3d
import pytest

from farfinder.head.geometry import CameraPose,Surface
from farfinder.head.intrinsics import BoardSpec,Intrinsics
from farfinder.head.scale import MetricScale,validate_reference,scale_from_board
from farfinder.head.session import HeadSession,utc_now
from farfinder.head.measurements import measure,calculate_measurements,save_measurements


def test_metric_scale_requires_independently_measured_reference():
    assert not validate_reference(2,20,0.1).valid
    scale=validate_reference(2,20,0.1,independently_measured=True)
    assert scale.valid
    assert scale.cm_per_unit==10
    assert MetricScale.from_dict(scale.to_dict())==scale


@pytest.mark.parametrize("kwargs",[dict(measured_cm=-1),dict(uncertainty_cm=4),dict(views=1),dict(reprojection_rms_px=5)])
def test_bad_reference_cannot_enable_centimeters(kwargs):
    values=dict(reconstruction_length=2,measured_cm=20,uncertainty_cm=0.1,independently_measured=True)
    values.update(kwargs)
    assert not validate_reference(**values).valid


def test_metric_certificate_rejects_nonfinite_scale():
    with pytest.raises(ValueError): MetricScale(True,float("nan"),0.01,"independently_measured_dimension",reference_count=1)


def test_static_board_validates_scale_using_registered_corners(tmp_path):
    spec=BoardSpec()
    k=np.array([[500.,0,320],[0,500,240],[0,0,1]])
    intrinsic=Intrinsics(640,480,k.tolist(),[0]*5,0,[0]*20,asdict(spec),utc_now())
    session=HeadSession.create(tmp_path,intrinsic)
    # Reconstruction coordinate units are 1/10 the known board cm dimensions.
    known=spec.board().getChessboardCorners()/10
    known[:,0]-=0.6
    known[:,1]-=0.4
    known[:,2]+=4
    cameras={}
    for i,center in enumerate((-1,0,1)):
        name=f"view{i}.png"
        camera=CameraPose(name,k,np.eye(3),[-center,0,0],np.zeros(5),640,480)
        cameras[name]=camera
        pixels,depths=camera.project(known)
        session.metadata["frames"].append(dict(image=name,board_observation=dict(ids=list(range(len(known))),pixels=pixels.tolist())))
    assert not scale_from_board(session,cameras).valid
    scale=scale_from_board(session,cameras,2.0,0.01)
    assert scale.valid
    assert scale.cm_per_unit==pytest.approx(10,rel=1e-5)
    assert scale.reference_count>=8
    # Board motion breaks rigid multi-view reprojection rather than changing scale silently.
    for pixel in session.metadata["frames"][-1]["board_observation"]["pixels"]:
        pixel[1]+=30
    assert not scale_from_board(session,cameras,2.0,0.01).valid


def test_measurements_stay_in_units_without_scale_and_missing_anatomy_unavailable():
    surface=Surface([[0,0,0],[3,0,0],[0,4,0]],[[0,1,2]])
    landmarks={"G":dict(status="DETECTED",point_3d=[0,0,0],uncertainty_units=0.01),
               "O":dict(status="DETECTED",point_3d=[3,0,0],uncertainty_units=0.01)}
    result=measure("G_to_O","G","O",landmarks,surface,MetricScale(),"euclidean")
    assert result["distance_value"]==3
    assert result["distance_unit"]=="reconstruction_units"
    assert not result["metric_scale_valid"]
    scale=validate_reference(1,10,0.01,True)
    assert measure("G_to_O","G","O",landmarks,surface,scale,"euclidean")["distance_value"]==30
    assert measure("V_to_P","V","P",landmarks,surface,scale,"euclidean")["quality_status"]=="UNAVAILABLE"
    results=calculate_measurements(landmarks,surface,scale)
    assert len(results)==18
    assert all(row["distance_value"] is None for row in results if row["measurement_name"].startswith("head_"))


def test_axis_extent_is_not_diagonal_distance_and_requires_observed_extrema():
    mesh=o3d.geometry.TriangleMesh.create_sphere(radius=2,resolution=24)
    vertices=np.asarray(mesh.vertices).copy()
    vertices[:,1]*=1.5
    surface=Surface(vertices,np.asarray(mesh.triangles))
    frame=dict(origin=[0,0,0],basis=np.eye(3).tolist())
    results=calculate_measurements({},surface,MetricScale(),frame,True)
    width=next(row for row in results if row["measurement_name"]=="head_width" and row["distance_type"]=="anatomical_axis_extent")
    assert width["distance_value"]==pytest.approx(4)
    assert width["quality_status"]=="ESTIMATED"


def test_measurement_files_have_required_schema(tmp_path):
    session=HeadSession.create(tmp_path)
    surface=Surface([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2]])
    results=calculate_measurements({},surface,MetricScale())
    save_measurements(session,results)
    assert (session.path/"measurements.csv").is_file()
    assert (session.path/"measurements.json").is_file()
    assert all("confidence_or_uncertainty" in row and "quality_status" in row for row in results)
