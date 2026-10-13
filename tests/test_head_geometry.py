import math

import numpy as np
import open3d as o3d
import pytest

from farfinder.head.geometry import CameraPose, Surface, triangulate, plane_pair_intersections
from farfinder.head.measurements import euclidean_distance


def cameras():
    k=np.array([[500.,0,320],[0,500,240],[0,0,1]])
    return {name:CameraPose(name,k,np.eye(3),[-center,0,0],np.zeros(5),640,480)
            for name,center in (("a",-2),("b",0),("c",2))}


def sphere(radius=10):
    mesh=o3d.geometry.TriangleMesh.create_sphere(radius=radius,resolution=16)
    return mesh,Surface(np.asarray(mesh.vertices),np.asarray(mesh.triangles))


def test_coordinate_transform_projection_and_ray():
    camera=cameras()["a"]
    point=np.array([0.2,0.1,8])
    assert np.allclose(camera.to_camera([point])[0],point+[2,0,0])
    pixel,depth=camera.project([point])
    origin,direction=camera.ray(pixel[0])
    assert np.dot(direction,(point-origin)/np.linalg.norm(point-origin))==pytest.approx(1)
    assert depth[0]==8
    assert CameraPose.from_dict(camera.to_dict()).center==pytest.approx(camera.center)


def test_multi_view_registration_recovers_true_3d_point():
    known=np.array([0.4,0.2,8])
    views=cameras()
    observations=[dict(image=name,pixel=camera.project([known])[0][0].tolist()) for name,camera in views.items()]
    reconstructed,quality=triangulate(observations,views,min_views=3)
    assert reconstructed==pytest.approx(known,abs=1e-8)
    assert quality["views"]==3
    assert quality["reprojection_rms_px"]<1e-6
    assert quality["uncertainty_units"]>0


def test_registration_rejects_inconsistent_pixels_and_tiny_baseline():
    views=cameras()
    known=[0.4,0.2,8]
    observations=[dict(image=name,pixel=camera.project([known])[0][0].tolist()) for name,camera in views.items()]
    observations[-1]["pixel"][1]+=40
    with pytest.raises(ValueError,match="reprojection"):
        triangulate(observations,views)
    one=views["a"]
    same={"a":one,"b":CameraPose("b",one.K,one.R,one.t,one.distortion,640,480)}
    pixel=one.project([known])[0][0].tolist()
    with pytest.raises(ValueError):
        triangulate([dict(image="a",pixel=pixel),dict(image="b",pixel=pixel)],same)


def test_euclidean_distance():
    assert euclidean_distance([0,0,0],[3,4,0])==5
    with pytest.raises(ValueError): euclidean_distance([0,0,float("nan")],[1,1,1])


def test_surface_edge_path_is_labeled_approximation_on_curved_strip():
    count=16
    angles=np.linspace(0,np.pi/2,count+1)
    vertices=np.array([[np.cos(angle),y,np.sin(angle)] for y in (0,0.1) for angle in angles])
    triangles=[]
    for i in range(count):
        j=i+count+1
        triangles.extend([[i,i+1,j],[i+1,j+1,j]])
    surface=Surface(vertices,triangles)
    distance,path,quality=surface.shortest_path(vertices[0],vertices[count])
    assert math.sqrt(2)<distance<math.pi/2
    assert distance==pytest.approx(2*count*np.sin(np.pi/(4*count)))
    assert "approximation" in quality["algorithm"]


def test_disconnected_mesh_does_not_produce_surface_distance():
    vertices=[[0,0,0],[1,0,0],[0,1,0],[10,0,0],[11,0,0],[10,1,0]]
    surface=Surface(vertices,[[0,1,2],[3,4,5]])
    with pytest.raises(ValueError,match="disconnected"):
        surface.shortest_path(vertices[0],vertices[3])


def test_plane_pair_intersects_real_mesh_and_pixel_requires_real_surface():
    mesh,surface=sphere()
    hits=plane_pair_intersections(surface,[1,0,0],0,[0,0,1],0)
    assert max(point[1] for point in hits)==pytest.approx(10)
    camera=CameraPose("front",[[500,0,320],[0,500,240],[0,0,1]],np.diag([1,-1,-1]),[0,0,30],np.zeros(5),640,480)
    assert surface.associate_pixel(camera,[320,240])==pytest.approx([0,0,10])
    with pytest.raises(ValueError,match="does not intersect"):
        surface.associate_pixel(camera,[0,0])


@pytest.mark.parametrize("rotation",[np.zeros((3,3)),np.diag([-1,1,1]),np.eye(3)*2])
def test_invalid_camera_rotations_rejected(rotation):
    with pytest.raises(ValueError):
        CameraPose("bad",np.eye(3),rotation,[0,0,0],np.zeros(5),640,480)
