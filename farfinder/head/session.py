"""Versioned session files. Pixel observations never implicitly become 3D."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4

import cv2
import numpy as np

from farfinder.head.capture_mode import capture_assumptions


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False,
                                         encoding="utf-8", suffix=".tmp") as file:
            temporary = file.name
            json.dump(payload, file, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


class HeadSession:
    def __init__(self, path, metadata):
        self.path = Path(path)
        self.metadata = metadata

    @classmethod
    def create(cls, data_dir, intrinsics=None, distance_calibration=None, capture_mode="moving-camera"):
        assumptions = capture_assumptions(capture_mode)
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:10]
        path = Path(data_dir) / "head_sessions" / name
        for directory in ("images", "masks", "segmentation"):
            (path / directory).mkdir(parents=True, exist_ok=False)
        session = cls(path, dict(schema_version=1, session_id=name, created_at=utc_now(),
                                state="CAPTURING", frames=[], selections=[],
                                intrinsics=intrinsics.to_dict() if intrinsics else None,
                                distance_calibration=distance_calibration.to_dict() if distance_calibration else None,
                                **assumptions,
                                metric_scale=dict(valid=False, reason="No independently measured reference registered")))
        session.save()
        return session

    @classmethod
    def load(cls, path):
        path = Path(path)
        payload = json.loads((path / "session.json").read_text())
        if payload.get("schema_version") != 1 or not isinstance(payload.get("frames"), list):
            raise ValueError("Unsupported head session schema")
        capture_assumptions(payload.get("capture_mode", "moving-camera"))
        names = set()
        for frame in payload["frames"]:
            name = frame["image"]
            if Path(name).name != name or name in names:
                raise ValueError("Invalid or duplicate session image name")
            names.add(name)
            if frame["width"] <= 0 or frame["height"] <= 0:
                raise ValueError("Invalid image dimensions")
        for point in payload.get("selections", []):
            if point.get("name") not in ("G", "O"):
                raise ValueError("Only optional G and O selections are allowed")
        return cls(path, payload)

    def save(self):
        save_json(self.path / "session.json", self.metadata)

    def add_frame(self, image, mask, categories, quality, orientation=None, board_observation=None):
        height, width = image.shape[:2]
        if mask.shape != (height, width) or categories.shape != (height, width):
            raise ValueError("Masks must have the original image dimensions")
        intrinsic = self.metadata.get("intrinsics")
        if intrinsic and (width,height)!=(intrinsic["width"],intrinsic["height"]):
            raise ValueError("Captured image does not match intrinsic calibration resolution")
        if self.metadata["frames"] and (width, height) != (self.metadata["frames"][0]["width"],
                                                         self.metadata["frames"][0]["height"]):
            raise ValueError("Camera resolution changed during capture; start a new session")
        name = f"frame_{len(self.metadata['frames']):05d}.png"
        # COLMAP masks are named like the image, with an additional .png suffix.
        for path, array in ((self.path / "images" / name, image),
                            (self.path / "masks" / (name + ".png"), mask),
                            (self.path / "segmentation" / name, categories)):
            if not cv2.imwrite(str(path), array):
                raise OSError(f"Could not save {path}")
        record = dict(image=name, timestamp_utc=utc_now(), width=width, height=height,
                      quality=quality, estimated_head_orientation=orientation,
                      camera_pose=None, camera_calibration=self.metadata["intrinsics"],
                      board_observation=board_observation)
        self.metadata["frames"].append(record)
        self.save()
        return name

    def stop(self):
        self.metadata["state"] = "CAPTURED"
        self.save()

    def select(self, name, image, pixel):
        if name not in ("G", "O"):
            raise ValueError("Only G (glabella) and O (posterior reference) can be selected")
        frame = next((f for f in self.metadata["frames"] if f["image"] == image), None)
        point = np.asarray(pixel, dtype=float)
        if (frame is None or point.shape != (2,) or not np.isfinite(point).all()
                or not 0 <= point[0] < frame["width"] or not 0 <= point[1] < frame["height"]):
            raise ValueError("Selection must be inside a captured image")
        item = dict(name=name, image=image, pixel=point.tolist(), timestamp_utc=utc_now(),
                    status="OBSERVATION_2D_ONLY", point_3d=None,
                    reason="Camera registration and validated surface association required")
        self.metadata["selections"] = [s for s in self.metadata["selections"]
                                       if not (s["name"] == name and s["image"] == image)] + [item]
        self.save()
        return item
