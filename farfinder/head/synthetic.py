"""Reproducible, textured ellipsoid images for real COLMAP validation.

Ground-truth poses are saved for evaluation only; reconstruction is not given
those poses, correspondences, or a synthetic mesh. No physical accuracy claim.
"""

from dataclasses import asdict
import json

import cv2
import numpy as np

from farfinder.head.geometry import CameraPose, camera_relative_to_object
from farfinder.head.intrinsics import BoardSpec, Intrinsics
from farfinder.head.session import HeadSession, save_json, utc_now


def look_at(center):
    center = np.asarray(center, float)
    forward = -center/np.linalg.norm(center)
    right = np.cross(forward, [0, 1, 0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack([right, down, forward])
    return rotation, -rotation @ center


def texture_map(seed=13):
    rng = np.random.default_rng(seed)
    texture = rng.integers(40, 220, (1024, 2048, 3), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (0, 0), 3)
    for i in range(2000):
        x, y = int(rng.integers(2048)), int(rng.integers(1024))
        color = tuple(int(c) for c in rng.integers(20, 235, 3))
        radius = int(rng.integers(5, 18))
        cv2.circle(texture, (x, y), radius, color, 2 if i % 3 else -1, cv2.LINE_AA)
    return texture


def render_ellipsoid(camera, radii, texture, background=None):
    y, x = np.indices((camera.height, camera.width))
    directions_camera = np.stack([(x-camera.K[0, 2])/camera.K[0, 0],
                                 (y-camera.K[1, 2])/camera.K[1, 1], np.ones_like(x)], axis=-1)
    directions = directions_camera @ camera.R
    center = camera.center
    inverse_squared = 1/np.square(radii)
    a = np.sum(directions**2 * inverse_squared, axis=-1)
    b = 2*np.sum(directions*center*inverse_squared, axis=-1)
    c = np.sum(center**2*inverse_squared)-1
    discriminant = b*b-4*a*c
    hit = discriminant > 0
    distance = (-b-np.sqrt(np.maximum(discriminant, 0)))/(2*a)
    hit &= distance > 0
    points = center+directions*distance[:, :, None]
    normalized = points/np.asarray(radii)
    longitude = np.arctan2(normalized[:, :, 2], normalized[:, :, 0])
    latitude = np.arccos(np.clip(normalized[:, :, 1], -1, 1))
    map_x = ((longitude+np.pi)/(2*np.pi)*texture.shape[1]).astype(np.float32)
    map_y = (latitude/np.pi*(texture.shape[0]-1)).astype(np.float32)
    color = cv2.remap(texture, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    if background is not None and (background.shape!=color.shape or background.dtype!=np.uint8):
        raise ValueError("Synthetic background must match the camera's uint8 RGB image dimensions")
    image = np.full(color.shape, 25, np.uint8) if background is None else background.copy()
    image[hit] = color[hit]
    mask = cv2.erode(hit.astype(np.uint8)*255, np.ones((5, 5), np.uint8))
    return image, mask


def generate_session(data_dir, orbit_views=24, capture_mode="moving-camera"):
    if orbit_views < 16 or orbit_views > 80:
        raise ValueError("Synthetic orbit needs 16-80 views")
    k = np.array([[700., 0, 320], [0, 700, 240], [0, 0, 1]])
    intrinsic = Intrinsics(640, 480, k.tolist(), [0]*5, 0, [0]*20, asdict(BoardSpec()), utc_now(), "synthetic")
    session = HeadSession.create(data_dir, intrinsic,capture_mode=capture_mode)
    radii = np.array([8., 12., 10.])
    texture = texture_map()
    truth = {}
    object_motion = {}
    fixed_rotation, fixed_translation = look_at([0,0,48])
    fixed_camera = CameraPose("",k,fixed_rotation,fixed_translation,np.zeros(5),640,480)
    background = cv2.resize(texture_map(seed=91),(640,480)) if capture_mode=="head-rotation" else None
    angles = [(angle, 0) for angle in np.linspace(0, 360, orbit_views, endpoint=False)]
    angles += [(angle, elevation) for elevation in (15,30,45,-15,-30,-45)
               for angle in (0,90,180,270)]
    for angle, elevation in angles:
        yaw, pitch = np.radians(angle), np.radians(elevation)
        center = 48*np.array([np.sin(yaw)*np.cos(pitch), np.sin(pitch), np.cos(yaw)*np.cos(pitch)])
        rotation, translation = look_at(center)
        camera = CameraPose("", k, rotation, translation, np.zeros(5), 640, 480)
        if capture_mode=="head-rotation":
            # Render a rigid target moving in front of ONE physical camera.
            # Effective head-relative poses are evaluation ground truth only.
            object_rotation = fixed_camera.R.T @ rotation
            object_translation = fixed_camera.R.T @ (translation-fixed_camera.t)
            camera = camera_relative_to_object(fixed_camera,object_rotation,object_translation)
        image, mask = render_ellipsoid(camera, radii, texture, background=background)
        categories = (mask > 0).astype(np.uint8)  # Ground-truth synthetic subject mask, not a model claim.
        name = session.add_frame(image, mask, categories,
                                 dict(status="ACCEPTED", source="synthetic renderer ground-truth mask"),
                                 dict(yaw_deg=float((angle+180)%360-180), pitch_deg=float(elevation), roll_deg=0,
                                      status="SYNTHETIC_GROUND_TRUTH"))
        camera.image = name
        truth[name] = camera.to_dict()
        if capture_mode=="head-rotation":
            object_motion[name] = dict(rotation=object_rotation.tolist(),translation=object_translation.tolist())
    session.metadata["synthetic_validation_only"] = True
    session.stop()
    save_json(session.path / "synthetic_ground_truth.json", dict(radii=radii.tolist(), cameras=truth,
              capture_mode=capture_mode,object_motion=object_motion,
              fixed_physical_camera=fixed_camera.to_dict() if capture_mode=="head-rotation" else None,
              unit="synthetic model units; not an independently measured physical centimeter reference"))
    return session


def validate_reconstruction(session):
    from farfinder.head.pipeline import load_geometry
    from farfinder.head.measurements import calculate_measurements, save_measurements
    from farfinder.head.scale import MetricScale
    mesh, surface, cameras, report = load_geometry(session)
    truth = json.loads((session.path / "synthetic_ground_truth.json").read_text())
    names = [name for name in cameras if name in truth["cameras"]]
    a = np.array([cameras[name].center for name in names])
    b = np.array([truth["cameras"][name]["center"] for name in names])
    x, y = a-a.mean(axis=0), b-b.mean(axis=0)
    u, singular, vt = np.linalg.svd(y.T @ x / len(x))
    sign = np.eye(3)
    sign[2, 2] = np.linalg.det(u @ vt)
    rotation = u @ sign @ vt
    scale = float(np.sum(singular*np.diag(sign))/np.mean(np.sum(x*x, axis=1)))
    translation = b.mean(axis=0)-scale*rotation @ a.mean(axis=0)
    transformed = scale * surface.vertices @ rotation.T+translation
    radii = np.array(truth["radii"])
    normalized_radii = np.linalg.norm(transformed/radii, axis=1)
    camera_errors = np.linalg.norm(scale*a @ rotation.T+translation-b, axis=1)
    validation = dict(status="SYNTHETIC_ONLY", registered_views=len(cameras),
        capture_mode=session.metadata.get("capture_mode","moving-camera"),
        camera_center_fit_rms_model_units=float(np.sqrt(np.mean(camera_errors**2))),
        median_surface_relative_radius_error=float(np.median(np.abs(normalized_radii-1))),
        reconstructed_extents_model_units=np.ptp(transformed, axis=0).tolist(), expected_extents_model_units=(2*radii).tolist(),
        note="True COLMAP feature extraction/matching/SfM and stereo; ground truth used only after reconstruction. "
             "Rigid synthetic target validation does not establish human-head rigidity, real-head or centimeter accuracy.")
    extent_errors = np.abs(np.ptp(transformed,axis=0)/(2*radii)-1)
    validation["extent_relative_errors"] = extent_errors.tolist()
    if (len(cameras) < len(truth["cameras"])*0.65 or validation["median_surface_relative_radius_error"] > 0.15
            or np.max(extent_errors) > 0.1):
        raise ValueError(f"Synthetic reconstruction quality failed: {validation}")
    report["synthetic_validation"] = validation
    save_json(session.path / "quality_report.json", report)
    # Demonstrate dimensions and mesh-edge measurements in reconstruction units,
    # with an explicitly synthetic orientation frame, without enabling cm output.
    frame = dict(origin=((-translation) @ rotation/scale).tolist(), basis=rotation.T.tolist(),
                 status="SYNTHETIC_GROUND_TRUTH_ONLY")
    results = calculate_measurements({}, surface, MetricScale(), frame, True)
    for item in results:
        if item["measurement_name"].startswith("head_"):
            item["measurement_name"] = "synthetic_ellipsoid_"+item["measurement_name"][5:]
            item["reason"] = "Synthetic geometry validation only; orientation evaluated from withheld ground truth"
    save_measurements(session, results)
    save_json(session.path / "synthetic_validation.json", validation)
    return validation
