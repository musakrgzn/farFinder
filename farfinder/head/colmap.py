"""COLMAP 3.8 CLI adapter with fixed calibrated intrinsics and head-only masks."""

from pathlib import Path
import os
import subprocess
import threading

import cv2
import numpy as np

from farfinder.head.geometry import CameraPose


class ReconstructionError(RuntimeError):
    pass


def quaternion_rotation(q):
    q = np.asarray(q, float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1e-3:
        raise ValueError("Invalid COLMAP rotation quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                     [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                     [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])


def read_text_model(directory):
    directory = Path(directory)
    cameras = {}
    for line in (directory / "cameras.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        camera_id, model, width, height = int(fields[0]), fields[1], int(fields[2]), int(fields[3])
        params = np.asarray(fields[4:], float)
        if model == "FULL_OPENCV" and len(params) == 12:
            fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = params
            if max(abs(k4), abs(k5), abs(k6)) > 1e-9:
                raise ReconstructionError("Rational distortion not supported by this calibrated stereo path")
            distortion = np.array([k1, k2, p1, p2, k3])
        elif model == "PINHOLE" and len(params) == 4:
            fx, fy, cx, cy = params
            distortion = np.zeros(5)
        elif model == "OPENCV" and len(params) == 8:
            fx, fy, cx, cy, k1, k2, p1, p2 = params
            distortion = np.array([k1, k2, p1, p2, 0])
        else:
            raise ReconstructionError(f"Unsupported calibrated camera model: {model}")
        k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        cameras[camera_id] = k, distortion, width, height
    poses, tracks = {}, {}
    lines = (directory / "images.txt").read_text().splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=9)
        if len(fields) != 10:
            raise ReconstructionError("Invalid COLMAP image pose record")
        image_id = int(fields[0])
        rotation = quaternion_rotation([float(v) for v in fields[1:5]])
        translation = np.asarray(fields[5:8], float)
        camera_id, name = int(fields[8]), fields[9]
        if Path(name).name != name:
            raise ReconstructionError("Nested image paths are unsupported")
        k, distortion, width, height = cameras[camera_id]
        poses[name] = CameraPose(name, k, rotation, translation, distortion, width, height)
        observations = lines[index].split() if index < len(lines) else []
        index += 1
        if len(observations) % 3:
            raise ReconstructionError("Invalid COLMAP point observations")
        for offset in range(0, len(observations), 3):
            x, y, point_id = float(observations[offset]), float(observations[offset+1]), int(observations[offset+2])
            if point_id >= 0:
                tracks.setdefault(point_id, []).append(dict(image=name, pixel=[x, y], image_id=image_id))
    points, colors, errors, point_tracks = [], [], [], []
    for line in (directory / "points3D.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        point_id = int(fields[0])
        track = tracks.get(point_id, [])
        if len(track) < 3 or float(fields[7]) > 2.5:
            continue
        points.append([float(v) for v in fields[1:4]])
        colors.append([int(v)/255 for v in fields[4:7]])
        errors.append(float(fields[7]))
        point_tracks.append(track)
    return poses, np.asarray(points).reshape(-1, 3), np.asarray(colors).reshape(-1, 3), errors, point_tracks


class ColmapRunner:
    def __init__(self, executable="colmap", threads=4, timeout=900, cancel=None, progress=None):
        self.executable = executable
        self.threads = threads
        self.timeout = timeout
        self.cancel = cancel or threading.Event()
        self.progress = progress or (lambda message: None)

    def run(self, session, command, arguments):
        if self.cancel.is_set():
            raise ReconstructionError("Reconstruction cancelled")
        log_path = session.path / "colmap.log"
        environment = os.environ.copy()
        environment["QT_QPA_PLATFORM"] = "offscreen"
        environment.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
        environment.pop("QT_QPA_FONTDIR", None)
        self.progress(f"COLMAP: {command}")
        with log_path.open("a") as log:
            log.write("\n" + " ".join([self.executable, command, *arguments]) + "\n")
            log.flush()
            process = subprocess.Popen([self.executable, command, *arguments],
                                       stdout=log, stderr=log, env=environment)
            elapsed = 0
            while True:
                try:
                    code = process.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    elapsed += 0.5
                    if self.cancel.is_set() or elapsed > self.timeout:
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        raise ReconstructionError(f"COLMAP {command} cancelled/timed out; see colmap.log")
            if code:
                tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-8:])
                if command=="mapper" and "No good initial image pair" in tail:
                    raise ReconstructionError("COLMAP found no valid starting view pair. "
                        "Capture wider overlapping head viewpoints with stable visible features; see colmap.log")
                raise ReconstructionError(f"COLMAP {command} failed (exit {code}): {tail}")

    def reconstruct_sparse(self, session, intrinsic):
        frames = session.metadata["frames"]
        if len(frames) < 6:
            raise ReconstructionError("Need at least 6 sharp overlapping head frames; a full orbit usually needs 40-80")
        for frame in frames:
            intrinsic.check_resolution(frame["width"], frame["height"])
            if not (session.path / "masks" / (frame["image"] + ".png")).is_file():
                raise ReconstructionError("Missing subject mask. Background matching is not permitted.")
        workspace = session.path / "colmap"
        # Each run uses a new directory, avoiding stale databases/models after recapture.
        from uuid import uuid4
        workspace = workspace / uuid4().hex[:10]
        workspace.mkdir(parents=True)
        database = workspace / "database.db"
        sparse = workspace / "sparse"
        sparse.mkdir()
        k, d = intrinsic.K, intrinsic.D
        params = [k[0, 0], k[1, 1], k[0, 2], k[1, 2], *d[:4], d[4], 0, 0, 0]
        self.run(session, "feature_extractor", [
            "--database_path", str(database), "--image_path", str(session.path / "images"),
            "--ImageReader.mask_path", str(session.path / "masks"),
            "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "FULL_OPENCV",
            "--ImageReader.camera_params", ",".join(str(v) for v in params),
            "--SiftExtraction.use_gpu", "0", "--SiftExtraction.num_threads", str(self.threads),
            "--SiftExtraction.first_octave", "0", "--SiftExtraction.max_image_size", "1600",
            "--SiftExtraction.max_num_features", "6000"])
        self.run(session, "exhaustive_matcher", [
            "--database_path", str(database), "--SiftMatching.use_gpu", "0",
            "--SiftMatching.num_threads", str(self.threads), "--SiftMatching.guided_matching", "1"])
        self.run(session, "mapper", [
            "--database_path", str(database), "--image_path", str(session.path / "images"),
            "--output_path", str(sparse), "--Mapper.num_threads", str(self.threads),
            "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0",
            "--Mapper.ba_refine_extra_params", "0", "--Mapper.init_min_num_inliers", "40",
            "--Mapper.abs_pose_min_num_inliers", "30", "--Mapper.filter_max_reproj_error", "3",
            "--Mapper.filter_min_tri_angle", "3"])
        candidates = []
        for model in sorted(sparse.iterdir()):
            if not model.is_dir():
                continue
            text = model / "text"
            text.mkdir()
            self.run(session, "model_converter", ["--input_path", str(model), "--output_path", str(text),
                                                  "--output_type", "TXT"])
            parsed = read_text_model(text)
            candidates.append((len(parsed[0]), model, parsed))
        if not candidates:
            raise ReconstructionError("COLMAP could not register views. Improve head texture, overlap, and subject stationarity.")
        _, model, parsed = max(candidates, key=lambda item: item[0])
        return model, parsed, dict(disconnected_models=len(candidates))


def head_support(points, cameras, masks, minimum_views=2, minimum_fraction=0.75):
    """Exclude points not supported inside head masks in their visible camera views."""
    points = np.asarray(points, float)
    support = np.zeros(len(points), int)
    visible = np.zeros(len(points), int)
    for name, camera in cameras.items():
        projected, depths = camera.project(points)
        pixel = np.rint(projected).astype(int)
        valid = ((depths > 0) & (pixel[:, 0] >= 0) & (pixel[:, 0] < camera.width)
                 & (pixel[:, 1] >= 0) & (pixel[:, 1] < camera.height))
        indices = np.flatnonzero(valid)
        visible[indices] += 1
        mask = masks[name]
        inside = mask[pixel[indices, 1], pixel[indices, 0]] > 0
        support[indices[inside]] += 1
    keep = ((support >= minimum_views) & (support >= visible * minimum_fraction)
            & (support >= len(cameras)*0.6))
    return keep, support
