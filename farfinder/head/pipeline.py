"""Reconstruction orchestration and quality reports; failed stages retain evidence."""

from pathlib import Path
import threading

import cv2
import numpy as np
import open3d as o3d

from farfinder.head.colmap import ColmapRunner, ReconstructionError, head_support
from farfinder.head.capture import VIEW_NAMES
from farfinder.head.capture_mode import capture_assumptions, validate_head_turns
from farfinder.head.dense import cpu_stereo, observed_mesh
from farfinder.head.geometry import CameraPose, Surface
from farfinder.head.intrinsics import Intrinsics
from farfinder.head.scale import MetricScale, scale_from_board
from farfinder.head.session import save_json, utc_now


def classify_coverage(cameras, center, frames, pose_coordinate_frame="stationary_scene"):
    anchor = next((frame["image"] for frame in frames
                   if frame["image"] in cameras
                   and frame.get("estimated_head_orientation")
                   and abs(frame["estimated_head_orientation"].get("yaw_deg", 180)) < 15), None)
    if anchor is None:
        return dict(status="UNRESOLVED", reason="No frontal anatomical orientation anchor", views={}, complete=False)
    front_camera = cameras[anchor]
    front = front_camera.center-center
    front /= np.linalg.norm(front)
    up = -front_camera.R[1]
    up -= front * np.dot(up, front)
    up /= np.linalg.norm(up)
    left = np.cross(up, front)
    covered, orientations = set(), {}
    for name, camera in cameras.items():
        direction = camera.center-center
        direction /= np.linalg.norm(direction)
        elevation = float(np.degrees(np.arcsin(np.clip(np.dot(direction, up), -1, 1))))
        azimuth = float(np.degrees(np.arctan2(np.dot(direction, left), np.dot(direction, front)))) % 360
        view = "Top" if elevation > 35 else VIEW_NAMES[int(round(azimuth/45)) % 8]
        if elevation > -20:
            covered.add(view)
        source = ("Registered head-relative viewpoints; not physical webcam movement"
                  if pose_coordinate_frame=="head_relative" else "Registered camera orbit relative to frontal anchor")
        orientations[name] = dict(view=view, azimuth_deg=azimuth, elevation_deg=elevation,
                                  status="ESTIMATED", source=source)
    return dict(status="ESTIMATED", views=orientations, covered=sorted(covered),
                missing=[name for name in VIEW_NAMES if name not in covered],
                complete=all(name in covered for name in VIEW_NAMES),
                reason="Camera coverage does not prove all scalp geometry was reconstructed")


def load_geometry(session):
    mesh_path = session.path / "head_mesh.ply"
    if not mesh_path.is_file():
        raise ReconstructionError("No reconstructed mesh exists; run reconstruction first")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    surface = Surface(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    import json
    payload = json.loads((session.path / "camera_poses.json").read_text())
    cameras = {name: CameraPose.from_dict(item) for name, item in payload["cameras"].items()}
    report = json.loads((session.path / "quality_report.json").read_text())
    if report.get("reconstruction_status") != "SUCCEEDED":
        raise ReconstructionError("Mesh belongs to an unvalidated or failed reconstruction")
    return mesh, surface, cameras, report


def reconstruct(session, measured_square_cm=None, reference_uncertainty_cm=None,
                dense_backend="cpu", cancel=None, progress=None):
    cancel = cancel or threading.Event()
    progress = progress or print
    assumptions = capture_assumptions(session.metadata.get("capture_mode", "moving-camera"))
    pose_frame = assumptions["pose_coordinate_frame"]
    report = dict(schema_version=1, started_at=utc_now(), reconstruction_status="RUNNING",
                  metric_scale=MetricScale().to_dict(), scalp_surface_validated=False,
                  anatomical_accuracy="NOT_EVALUATED", centimeter_accuracy="NOT_EVALUATED",
                  subject_stationarity=assumptions["stationarity"], capture_mode=assumptions["capture_mode"],
                  experimental_capture=assumptions["experimental"], pose_coordinate_frame=pose_frame,
                  rigidity_status="NOT_INDEPENDENTLY_VERIFIED", stages={})
    session.metadata["state"] = "RECONSTRUCTING"
    session.metadata["metric_scale"] = MetricScale(reason="New reconstruction has not been validated").to_dict()
    for frame in session.metadata["frames"]:
        frame["camera_pose"] = None
    for selection in session.metadata.get("selections",[]):
        selection.update(status="OBSERVATION_2D_ONLY",point_3d=None,
                         reason="New reconstruction requires a new registered camera/surface association")
    session.save()
    save_json(session.path / "quality_report.json", report)
    # Invalidate published results at the beginning of a new run. Existing mesh
    # files can remain as evidence, but FAILED/RUNNING reports cannot load them.
    from farfinder.head.landmarks import unresolved_landmarks
    from farfinder.head.measurements import calculate_measurements, save_measurements
    empty_landmarks = unresolved_landmarks()
    save_json(session.path / "camera_poses.json",dict(schema_version=1,status="UNRESOLVED",
                                                  pose_coordinate_frame=pose_frame,cameras={}))
    save_json(session.path / "landmarks.json",dict(schema_version=1,landmarks=empty_landmarks,observations=[],
                                                  hairline=dict(status="UNRESOLVED")))
    save_measurements(session,calculate_measurements(empty_landmarks,None,MetricScale()))
    try:
        intrinsic_payload = session.metadata.get("intrinsics")
        if not intrinsic_payload:
            raise ReconstructionError("Intrinsic calibration is required. Face-size calibration is not a substitute.")
        intrinsic = Intrinsics(**intrinsic_payload)
        if assumptions["capture_mode"]=="head-rotation":
            report["stages"]["head_turn_capture"] = validate_head_turns(session.metadata["frames"])
        runner = ColmapRunner(cancel=cancel, progress=progress)
        model, parsed, sparse_info = runner.reconstruct_sparse(session, intrinsic)
        cameras, points, colors, errors, tracks = parsed
        fraction = len(cameras)/len(session.metadata["frames"])
        masks = {name: cv2.imread(str(session.path / "masks" / (name+".png")), cv2.IMREAD_GRAYSCALE)
                 for name in cameras}
        if any(mask is None for mask in masks.values()):
            raise ReconstructionError("Missing head masks for registered images")
        if len(points) < 150 or len(cameras) < 6 or fraction < 0.65:
            raise ReconstructionError(f"Insufficient SfM: {len(cameras)} registered views ({fraction:.0%}), {len(points)} reliable points")
        keep, support = head_support(points, cameras, masks)
        points, colors = points[keep], colors[keep]
        if len(points) < 150 or np.mean(keep) < 0.6:
            raise ReconstructionError("Reconstruction is not adequately supported by subject masks; background leakage or poor head geometry")
        report["stages"]["sfm"] = sparse_info | dict(registered_views=len(cameras), registered_fraction=fraction,
            reliable_points=len(points), mean_reprojection_error_px=float(np.mean(errors)),
            subject_mask_supported_fraction=float(np.mean(keep)), model_path=str(model.relative_to(session.path)),
            feature_matching="Head-only masked SIFT; COLMAP geometric verification; fixed calibrated intrinsics")
        serialized_poses = {name: pose.to_dict() | {"pose_coordinate_frame":pose_frame}
                            for name, pose in cameras.items()}
        save_json(session.path / "camera_poses.json", dict(schema_version=1, geometry_unit="reconstruction_units",
                  pose_coordinate_frame=pose_frame,cameras=serialized_poses))
        for frame in session.metadata["frames"]:
            if frame["image"] in cameras:
                frame["camera_pose"] = serialized_poses[frame["image"]]
        report["coverage"] = classify_coverage(cameras, np.median(points, axis=0), session.metadata["frames"],pose_frame)
        cloud = None
        if dense_backend == "cuda":
            # This path is available with a CUDA-enabled replacement COLMAP binary/image.
            # The supplied Debian image is CPU-only; report capability failure explicitly.
            import os
            import subprocess
            environment = os.environ.copy()
            environment["QT_QPA_PLATFORM"] = "offscreen"
            environment.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
            capability = subprocess.run(["colmap", "patch_match_stereo", "-h"], env=environment,
                                        capture_output=True, text=True)
            if capability.returncode or not Path("/dev/nvidia0").exists():
                report["stages"]["cuda_dense"] = dict(status="UNAVAILABLE", reason="CUDA COLMAP and an exposed NVIDIA device required; using CPU stereo")
            else:
                dense = model.parent.parent / "dense"
                dense.mkdir()
                runner.run(session, "image_undistorter", ["--image_path", str(session.path / "images"),
                    "--input_path", str(model), "--output_path", str(dense), "--output_type", "COLMAP", "--max_image_size", "1600"])
                runner.run(session, "patch_match_stereo", ["--workspace_path", str(dense),
                    "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "true"])
                runner.run(session, "stereo_fusion", ["--workspace_path", str(dense), "--workspace_format", "COLMAP",
                    "--input_type", "geometric", "--output_path", str(dense / "fused.ply")])
                cloud = o3d.io.read_point_cloud(str(dense / "fused.ply"))
                selected, _ = head_support(np.asarray(cloud.points), cameras, masks)
                cloud = cloud.select_by_index(np.flatnonzero(selected).tolist())
                report["stages"]["dense"] = dict(backend="COLMAP CUDA", points=len(cloud.points))
        if cloud is None:
            cloud, dense_info = cpu_stereo(session, cameras, masks, points, cancel, progress)
            report["stages"]["dense"] = dense_info
        if cloud is None or len(cloud.points) < 500:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            cloud.colors = o3d.utility.Vector3dVector(colors)
            report["stages"]["dense"]["fallback"] = "Sparse observed head points; low surface detail"
        if cancel.is_set():
            raise ReconstructionError("Reconstruction cancelled")
        progress("Open3D: processing observed head surface")
        if not o3d.io.write_point_cloud(str(session.path / "pointcloud.ply"), cloud):
            raise ReconstructionError("Could not save head point cloud")
        cloud, mesh, mesh_report = observed_mesh(cloud)
        report["stages"]["surface"] = mesh_report
        if mesh_report["largest_component_fraction"] < 0.5:
            raise ReconstructionError("Surface too fragmented; acquire better overlapping views")
        o3d.io.write_point_cloud(str(session.path / "pointcloud.ply"), cloud)
        if not o3d.io.write_triangle_mesh(str(session.path / "head_mesh.ply"), mesh):
            raise ReconstructionError("Could not save head mesh")
        scale = scale_from_board(session, cameras, measured_square_cm, reference_uncertainty_cm)
        report["metric_scale"] = scale.to_dict()
        report["reconstruction_status"] = "SUCCEEDED"
        report["finished_at"] = utc_now()
        report["limitations"] = ["Photogrammetry can fail on low-texture hair/skin or a moving subject",
            "Mesh represents observed outer hair/skin, not hidden scalp", "Coverage does not guarantee a complete mesh",
            "Anatomical accuracy and physical centimeter accuracy require ground-truth evaluation"]
        if assumptions["experimental"]:
            report["limitations"] += ["Fixed webcam head turns assume approximately rigid head geometry; expression/hair changes can invalidate it",
                "Head-relative poses are not physical camera motion; a stationary board cannot provide this reconstruction scale",
                "Front/side head turns cannot reconstruct unseen posterior or top surfaces"]
        session.metadata["metric_scale"] = scale.to_dict()
        session.metadata["state"] = "RECONSTRUCTED"
        save_json(session.path / "quality_report.json", report)
        session.save()
        # Every successful session gets explicit unavailable measurement/landmark records
        # even before the user invokes the anatomical detector.
        landmarks = unresolved_landmarks()
        save_json(session.path / "landmarks.json", dict(schema_version=1, landmarks=landmarks, observations=[], hairline={"status":"UNRESOLVED"}))
        save_measurements(session, calculate_measurements(landmarks, Surface(np.asarray(mesh.vertices), np.asarray(mesh.triangles)), scale))
        progress("Reconstruction complete: observed surface available; " + ("metric reference validated" if scale.valid else "arbitrary units"))
        return report
    except (OSError, ValueError, RuntimeError) as error:
        report["reconstruction_status"] = "FAILED"
        report["error"] = str(error)
        report["finished_at"] = utc_now()
        session.metadata["state"] = "FAILED"
        session.save()
        save_json(session.path / "quality_report.json", report)
        raise ReconstructionError(str(error)) from error
