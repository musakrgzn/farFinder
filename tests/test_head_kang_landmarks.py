import numpy as np
import open3d as o3d
import pytest

from farfinder.head.ears import EarDetector,EAR_NAMES,HeadbandPoleDetector,YoloOnnxEarDetector
from farfinder.head.geometry import Surface
from farfinder.head.kang import anatomical_frame,calculate_kang
from farfinder.head.landmarks import facial_references,hairline_boundary,unresolved_landmarks,associate_selections
from farfinder.head.session import HeadSession


def geometry():
    mesh=o3d.geometry.TriangleMesh.create_sphere(radius=10,resolution=24)
    return Surface(np.asarray(mesh.vertices),np.asarray(mesh.triangles))


def references():
    locations=dict(left_tragus=[8,0,0],right_tragus=[-8,0,0],left_superior_ear_root=[8,2,0],
                   right_superior_ear_root=[-8,2,0],left_eyebrow_peak=[2,4,7],right_eyebrow_peak=[-2,4,7],
                   nose_tip=[0,0,9],chin=[0,-6,7])
    return {name:dict(status="DETECTED",point_3d=point,uncertainty_units=0.05,confidence=0.95,
                     anatomical_validated=True,source="synthetic_ground_truth_only") for name,point in locations.items()}


def test_no_default_ear_or_pole_coordinates_are_invented():
    image=np.zeros((100,100,3),np.uint8)
    ears=EarDetector().detect(image)
    assert set(ears)==set(EAR_NAMES)
    assert all(item["pixel"] is None and item["status"]=="UNRESOLVED" for item in ears.values())
    assert all(item["pixel"] is None for item in HeadbandPoleDetector().detect(image).values())


def test_anatomical_frame_and_kang_planes_on_known_synthetic_surface():
    points=references()
    frame=anatomical_frame(points)
    assert np.array(frame["basis"])==pytest.approx(np.eye(3))
    kang,frame=calculate_kang(geometry(),points,scalp_surface_validated=True)
    assert kang["V"]["point_3d"]==pytest.approx([0,10,0],abs=0.05)
    assert kang["P"]["point_3d"][0]==pytest.approx(0,abs=1e-6)
    assert kang["P"]["point_3d"][1]==pytest.approx(2,abs=1e-6)
    assert kang["P"]["point_3d"][2]<-9
    assert kang["FL"]["status"]==kang["FR"]["status"]=="UNRESOLVED"
    assert kang["V"]["uncertainty_units"]>0


def test_kang_requires_actual_scalp_and_validated_bilateral_anatomy():
    points=references()
    kang,frame=calculate_kang(geometry(),points)
    assert all(item["status"]=="UNRESOLVED" for item in kang.values())
    points["left_tragus"]["anatomical_validated"]=False
    kang,frame=calculate_kang(geometry(),points,True)
    assert all("validated" in item["reason"] for item in kang.values())
    del points["left_tragus"]
    kang,frame=calculate_kang(geometry(),points,True)
    assert all(item["point_3d"] is None for item in kang.values())


def test_bad_bilateral_labels_are_rejected():
    points=references()
    points["left_tragus"],points["right_tragus"]=points["right_tragus"],points["left_tragus"]
    with pytest.raises(ValueError,match="inconsistent"):
        anatomical_frame(points)


def test_FL_FR_require_registered_hairline_and_geometric_poles():
    surface=geometry()
    points=references()
    hairline=[]
    for side,sign in (("left",1),("right",-1)):
        pole,_=surface.nearest_vertex([sign*6,6,5])
        brow,_=surface.nearest_vertex([sign*3,1,9])
        points[f"{side}_headband_pole"]=dict(status="DETECTED",point_3d=surface.vertices[pole].tolist(),
            uncertainty_units=0.05,confidence=0.95,anatomical_validated=True,source="synthetic_geometry_only")
        points[f"{side}_eyebrow_peak"]["point_3d"]=surface.vertices[brow].tolist()
        _,path,_=surface.shortest_path(surface.vertices[pole],surface.vertices[brow])
        hairline.append(path[len(path)//2].tolist())
    kang,frame=calculate_kang(surface,points,True,hairline=dict(status="ESTIMATED",points_3d=hairline))
    assert kang["FL"]["status"]==kang["FR"]["status"]=="ESTIMATED"
    assert "approximation" in kang["FL"]["tape_path"]["algorithm"]
    kang,frame=calculate_kang(surface,points,True,hairline=None)
    assert kang["FL"]["status"]==kang["FR"]["status"]=="UNRESOLVED"


def test_front_hairline_requires_a_real_hair_skin_boundary():
    points=np.full((478,2),[160,160],float)
    points[10]=[160,40]
    points[152]=[160,270]
    points[70]=[100,140]
    points[300]=[220,140]
    points[234]=[60,180]
    points[454]=[260,180]
    categories=np.zeros((320,320),np.uint8)
    categories[45:90,70:250]=1
    categories[90:200,70:250]=3
    boundary,reason=hairline_boundary(categories,points,(0,0,0))
    assert np.count_nonzero(boundary)>15
    assert not np.any(hairline_boundary(categories,points,(50,0,0))[0])
    assert not np.any(hairline_boundary(np.zeros_like(categories),points,(0,0,0))[0])


def test_optional_G_O_stay_2d_without_registered_mesh(tmp_path):
    session=HeadSession.create(tmp_path)
    image=np.zeros((100,100,3),np.uint8)
    session.add_frame(image,np.full((100,100),255,np.uint8),np.zeros((100,100),np.uint8),{})
    session.select("G","frame_00000.png",[50,50])
    landmarks=unresolved_landmarks()
    associate_selections(session,{},None,landmarks)
    assert landmarks["G"]["point_3d"] is None
    assert session.metadata["selections"][0]["status"]=="OBSERVATION_2D_ONLY"
    with pytest.raises(ValueError,match="Only G"):
        session.select("V","frame_00000.png",[50,50])
