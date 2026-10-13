"""Local facial/hairline detections and validated cross-view surface association."""

from pathlib import Path
import itertools
import json

import cv2
import mediapipe as mp
import numpy as np
import open3d as o3d

from farfinder.head.ears import EAR_NAMES, EarDetector, HeadbandPoleDetector
from farfinder.head.geometry import triangulate
from farfinder.head.kang import calculate_kang, unresolved
from farfinder.head.measurements import calculate_measurements, save_measurements
from farfinder.head.pipeline import load_geometry
from farfinder.head.scale import MetricScale
from farfinder.head.session import save_json
from farfinder.vision import pose_angles


NAMES = ["G", "O", "glabella_region", "left_eyebrow_peak", "right_eyebrow_peak", "nose_tip", "chin",
         *EAR_NAMES, "left_headband_pole", "right_headband_pole", "V", "P", "FL", "FR"]


def unresolved_landmarks():
    records = {name: unresolved("No multi-view anatomical detection and observed-surface registration") for name in NAMES}
    for name in EAR_NAMES:
        records[name] = unresolved("No validated local tragus/superior-ear-root keypoint model configured")
    for name in ("left_headband_pole", "right_headband_pole"):
        records[name] = unresolved("Headband pole position cannot be inferred from generic face shape")
    return records


def facial_references(points, pose=None):
    """Facial orientation/reference estimators, not Kang ear or scalp substitutes."""
    points = np.asarray(points, float)
    if len(points) < 468 or points.shape[1] != 2 or not np.isfinite(points).all():
        return {}
    roll = np.radians(pose[2]) if pose else 0
    c, s = np.cos(roll), np.sin(roll)
    unrolled = points @ np.array([[c, -s], [s, c]])
    left_indices, right_indices = [276, 283, 282, 295, 285], [46, 53, 52, 65, 55]
    left_peak = left_indices[int(np.argmin(unrolled[left_indices, 1]))]
    right_peak = right_indices[int(np.argmin(unrolled[right_indices, 1]))]
    glabella = (points[168]+points[9])/2
    references = {"G": glabella, "glabella_region": glabella,
                  "left_eyebrow_peak": points[left_peak], "right_eyebrow_peak": points[right_peak],
                  "nose_tip": points[1], "chin": points[152]}
    return {name: dict(pixel=pixel.tolist(), status="ESTIMATED", confidence=0.55,
                       visible=True, anatomical_validated=False, source="mediapipe_facial_reference_estimator",
                       reason="Semantic region/peak estimate; anatomical point accuracy not evaluated")
            for name, pixel in references.items()}


def hairline_boundary(categories, face_points, pose=None):
    """Conservative hair/face-skin boundary above the eyebrows in frontal views."""
    mask = np.zeros(categories.shape, np.uint8)
    if face_points is None or (pose and (abs(pose[0]) > 30 or abs(pose[1]) > 25)):
        return mask, "Frontal face visibility required for frontal hairline"
    points = np.asarray(face_points)
    height, width = categories.shape
    brow_y = float(min(points[70, 1], points[300, 1]))
    face_height = max(1, points[152, 1]-points[10, 1])
    x1 = max(0, int(min(points[234, 0], points[454, 0])))
    x2 = min(width, int(max(points[234, 0], points[454, 0]))+1)
    y1, y2 = max(0, int(brow_y-face_height*0.65)), min(height, int(brow_y-face_height*0.04))
    if x2 <= x1 or y2 <= y1:
        return mask, "Invalid forehead region"
    hair = (categories == 1).astype(np.uint8)
    skin = (categories == 3).astype(np.uint8)
    boundary = skin & cv2.dilate(hair, np.ones((5, 5), np.uint8))
    mask[y1:y2, x1:x2] = boundary[y1:y2, x1:x2]*255
    if np.count_nonzero(mask) < 15:
        return mask, "No visible frontal hair/skin boundary; baldness, occlusion, or segmentation failure"
    return mask, "ESTIMATED local hair/face segmentation boundary; hairline accuracy not evaluated"


def register_reference(observations, cameras, surface):
    if surface is None:
        raise ValueError("No validated reconstructed surface")
    usable = [item for item in observations if item["image"] in cameras and item.get("visible")]
    # RANSAC over camera pairs handles occasional model jitter without fusing
    # unrelated pixel coordinates: every accepted observation must reproject.
    best = None
    for pair in itertools.islice(itertools.combinations(usable, 2), 80):
        try:
            point, _ = triangulate(pair, cameras)
        except ValueError:
            continue
        inliers = []
        for item in usable:
            pixel, depth = cameras[item["image"]].project([point])
            if depth[0] > 0 and np.linalg.norm(pixel[0]-item["pixel"]) <= 3:
                inliers.append(item)
        if best is None or len(inliers) > len(best):
            best = inliers
    if best is None or len(best) < 3:
        raise ValueError("Need 3 geometrically consistent registered views of the same anatomical reference")
    point, quality = triangulate(best, cameras, min_views=3)
    vertex, snap = surface.nearest_vertex(point, surface.spacing*2)
    associated = surface.vertices[vertex]
    for item in best:
        hit = surface.associate_pixel(cameras[item["image"]], item["pixel"])
        if np.linalg.norm(hit-associated) > surface.spacing*3+quality["uncertainty_units"]:
            raise ValueError("Reference is occluded or inconsistent with the observed surface")
    quality["surface_snap_units"] = snap
    quality["uncertainty_units"] = float(np.hypot(quality["uncertainty_units"], snap))
    return associated, quality, best


def register_hairline(mesh, surface, cameras, boundaries):
    vertices = surface.vertices
    support = np.zeros(len(vertices), int)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    for name, boundary in boundaries.items():
        if name not in cameras or np.count_nonzero(boundary) < 15:
            continue
        camera = cameras[name]
        projected, depths = camera.project(vertices)
        pixels = np.rint(projected).astype(int)
        valid = (depths > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < camera.width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < camera.height)
        candidates = np.flatnonzero(valid)
        distance = cv2.distanceTransform(255-boundary, cv2.DIST_L2, 3)
        candidates = candidates[distance[pixels[candidates, 1], pixels[candidates, 0]] <= 3]
        if not len(candidates):
            continue
        directions = vertices[candidates]-camera.center
        expected = np.linalg.norm(directions, axis=1)
        directions /= expected[:, None]
        rays = np.column_stack([np.broadcast_to(camera.center, directions.shape), directions]).astype(np.float32)
        hits = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        visible = np.isfinite(hits) & (np.abs(hits-expected) <= surface.spacing*2)
        support[candidates[visible]] += 1
    selected = np.flatnonzero(support >= 2)
    if len(selected) < 10:
        return dict(status="UNRESOLVED", points_3d=[], confidence=None,
                    reason="Hairline boundary lacks consistent visibility in at least two registered views")
    # These are the same reconstructed mesh vertices checked in multiple views,
    # not averages of unrelated 2D boundary positions.
    return dict(status="ESTIMATED", points_3d=vertices[selected].tolist(), confidence=0.5,
                uncertainty_units=surface.spacing*2, visibility_views=support[selected].tolist(),
                reason="Observed surface vertices supported by frontal hairline masks in multiple registered views",
                accuracy="Hairline anatomical accuracy not evaluated")


def associate_selections(session, cameras, surface, landmarks):
    for item in session.metadata.get("selections", []):
        name = item["name"]
        if name not in ("G", "O"):
            raise ValueError("Only G and O may be manually selected")
        try:
            if surface is None or item["image"] not in cameras:
                raise ValueError("No registered camera and observed surface for this captured image")
            point = surface.associate_pixel(cameras[item["image"]], item["pixel"])
            item.update(status="DETECTED", point_3d=point.tolist(),
                        reason="Optional 2D selection ray intersects the registered observed surface")
            landmarks[name] = dict(status="DETECTED", point_3d=point.tolist(), visible=True, confidence=None,
                uncertainty_units=surface.spacing*2, anatomical_validated=False, source="optional_G_O_selection",
                reason="Surface association validated; manual anatomical selection accuracy is unknown")
        except ValueError as error:
            item.update(status="OBSERVATION_2D_ONLY", point_3d=None, reason=str(error))
            landmarks[name] = unresolved(f"Optional selection remains 2D only: {error}")
    session.save()


def detect_session(session, ear_detector=None, pole_detector=None, progress=None):
    ear_detector = ear_detector or EarDetector()
    pole_detector = pole_detector or HeadbandPoleDetector()
    progress = progress or print
    mesh, surface, cameras, report = None, None, {}, {}
    try:
        mesh, surface, cameras, report = load_geometry(session)
    except RuntimeError:
        pass
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path="/app/models/face_landmarker.task",
                                          delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=2,
        min_face_detection_confidence=0.6, min_face_presence_confidence=0.6,
        output_facial_transformation_matrixes=True)
    observations, boundaries = [], {}
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        for index, frame in enumerate(session.metadata["frames"]):
            progress(f"Local landmarks: frame {index+1}/{len(session.metadata['frames'])}")
            image = cv2.imread(str(session.path / "images" / frame["image"]))
            if image is None:
                continue
            height, width = image.shape[:2]
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
            if len(result.face_landmarks) == 1:
                points = np.array([(p.x*width, p.y*height) for p in result.face_landmarks[0]])
                pose = pose_angles(result.facial_transformation_matrixes[0]) if result.facial_transformation_matrixes else None
                if pose is None or abs(pose[0]) <= 45:
                    for name, item in facial_references(points, pose).items():
                        pixel = item["pixel"]
                        if 0 <= pixel[0] < width and 0 <= pixel[1] < height:
                            observations.append(dict(name=name, image=frame["image"], **item))
                categories = cv2.imread(str(session.path / "segmentation" / frame["image"]), cv2.IMREAD_GRAYSCALE)
                if categories is not None:
                    boundary, reason = hairline_boundary(categories, points, pose)
                    boundaries[frame["image"]] = boundary
                    observations.append(dict(name="frontal_hairline", image=frame["image"], status="ESTIMATED" if np.any(boundary) else "UNRESOLVED",
                                             confidence=0.5 if np.any(boundary) else None, visible=bool(np.any(boundary)), reason=reason,
                                             boundary_pixels=np.column_stack(np.where(boundary > 0)[::-1])[::3].tolist()))
            for detector_instance in (ear_detector, pole_detector):
                for name, item in detector_instance.detect(image).items():
                    observations.append(dict(image=frame["image"], **item))
    landmarks = unresolved_landmarks()
    for name in NAMES:
        if name in ("O", "V", "P", "FL", "FR"):
            continue
        candidates = [item for item in observations if item["name"] == name and item.get("pixel") is not None]
        if not candidates:
            continue
        try:
            point, quality, inliers = register_reference(candidates, cameras, surface)
            validated = all(item.get("anatomical_validated", False) for item in inliers)
            landmarks[name] = dict(status="DETECTED" if validated else "ESTIMATED", point_3d=point.tolist(),
                visible=True, confidence=float(np.median([item["confidence"] for item in inliers])),
                anatomical_validated=validated, source=inliers[0]["source"], uncertainty_units=quality["uncertainty_units"],
                registration_quality=quality, reason="Same semantic reference triangulated and associated with observed surface",
                accuracy="Anatomical accuracy not claimed; refer to detector evaluation provenance")
        except ValueError as error:
            landmarks[name] = unresolved(str(error))
    associate_selections(session, cameras, surface, landmarks)
    hairline = (register_hairline(mesh, surface, cameras, boundaries) if surface
                else dict(status="UNRESOLVED", points_3d=[], reason="No validated head surface"))
    frame = None
    if surface:
        kang, frame = calculate_kang(surface, landmarks, report.get("scalp_surface_validated", False), hairline=hairline)
        landmarks.update(kang)
    payload = dict(schema_version=1, landmarks=landmarks, observations=observations, hairline=hairline,
                   anatomical_frame=frame, anatomical_accuracy="NOT_EVALUATED",
                   paper_basis="User-supplied Kang definitions, Lee et al. 2011 DOI 10.1111/j.1524-4725.2011.02020.x; full protocol not independently reproduced")
    save_json(session.path / "landmarks.json", payload)
    scale = MetricScale.from_dict(report.get("metric_scale", {}))
    results = calculate_measurements(landmarks, surface, scale, frame, report.get("coverage", {}).get("complete", False))
    save_measurements(session, results)
    return payload
