"""Modular local ear detector. No generic Face Mesh point is an ear reference."""

from pathlib import Path
import hashlib
import json

import cv2
import numpy as np


EAR_NAMES = ["left_tragus", "right_tragus", "left_superior_ear_root", "right_superior_ear_root"]


def missing_ear(name, reason="No validated anatomical ear keypoint model configured"):
    return dict(name=name, status="UNRESOLVED", pixel=None, confidence=None,
                visible=False, anatomical_validated=False, source="ear_detector_interface", reason=reason)


class EarDetector:
    """Replaceable four-keypoint detector contract, with explicit missing outputs."""
    def detect(self, image):
        return {name: missing_ear(name) for name in EAR_NAMES}


class YoloOnnxEarDetector(EarDetector):
    """One-class YOLO pose ONNX with four keypoints, executed by OpenCV DNN.

    Expected decoded output: [1,17,N] or [1,N,17]: xywh, object score,
    four (x,y,confidence) keypoints. No pretrained weights are invented/bundled.
    The manifest must document the exact semantics and evaluation evidence.
    """
    def __init__(self, model_path, manifest_path):
        self.manifest = json.loads(Path(manifest_path).read_text())
        if self.manifest.get("keypoints") != EAR_NAMES or self.manifest.get("output_format") != "yolo_pose_1class_4keypoints":
            raise ValueError("Ear model manifest must specify the four anatomical ear keypoints and supported output format")
        self.input_size = int(self.manifest.get("input_size", 640))
        if not 128 <= self.input_size <= 2048:
            raise ValueError("Invalid ear model input size")
        self.threshold = float(self.manifest.get("confidence_threshold", 0.65))
        if not 0.5 <= self.threshold <= 1:
            raise ValueError("Ear confidence threshold must be between 0.5 and 1")
        checksum = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
        if self.manifest.get("sha256") != checksum:
            raise ValueError("Ear model SHA-256 does not match manifest")
        self.model = cv2.dnn.readNetFromONNX(str(model_path))
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        evaluation = self.manifest.get("evaluation", {})
        # A manifest alone cannot prove accuracy. Preserve the evaluation provenance
        # and require annotated data evidence before geometry uses the predictions.
        self.validated = (evaluation.get("annotated_samples", 0) >= 50
                          and evaluation.get("anatomical_labels_verified") is True
                          and isinstance(evaluation.get("report_sha256"), str)
                          and len(evaluation["report_sha256"]) == 64)
        if self.validated:
            report_path = Path(manifest_path).parent / evaluation.get("report_file", "")
            self.validated = (report_path.is_file() and
                hashlib.sha256(report_path.read_bytes()).hexdigest() == evaluation["report_sha256"])

    def detect(self, image):
        height, width = image.shape[:2]
        ratio = min(self.input_size/width, self.input_size/height)
        resized = cv2.resize(image, (round(width*ratio), round(height*ratio)))
        left, top = (self.input_size-resized.shape[1])//2, (self.input_size-resized.shape[0])//2
        padded = np.full((self.input_size, self.input_size, 3), 114, np.uint8)
        padded[top:top+resized.shape[0], left:left+resized.shape[1]] = resized
        self.model.setInput(cv2.dnn.blobFromImage(padded, 1/255, (self.input_size, self.input_size), swapRB=True))
        output = np.asarray(self.model.forward()).squeeze(axis=0)
        if output.ndim != 2:
            raise ValueError("Unsupported ear model output tensor")
        if output.shape[0] == 17:
            output = output.T
        if output.shape[1] != 17:
            raise ValueError("Ear model must return 17 values per detection")
        valid = output[np.isfinite(output).all(axis=1) & (output[:, 4] >= self.threshold)]
        if not len(valid):
            return {name: missing_ear(name, "Ear keypoint not visible or below confidence threshold") for name in EAR_NAMES}
        boxes = [[float(row[0]-row[2]/2), float(row[1]-row[3]/2), float(row[2]), float(row[3])] for row in valid]
        selected = np.asarray(cv2.dnn.NMSBoxes(boxes, valid[:, 4].tolist(), self.threshold, 0.45)).reshape(-1)
        if len(selected) != 1:
            return {name: missing_ear(name, "Ambiguous multiple-head ear detections") for name in EAR_NAMES}
        row = valid[selected[0]]
        records = {}
        for i, name in enumerate(EAR_NAMES):
            x, y, confidence = row[5+3*i:8+3*i]
            pixel = [(float(x)-left)/ratio, (float(y)-top)/ratio]
            visible = confidence >= self.threshold and 0 <= pixel[0] < width and 0 <= pixel[1] < height
            records[name] = dict(name=name, status="DETECTED" if visible else "UNRESOLVED",
                pixel=pixel if visible else None, confidence=float(confidence), visible=bool(visible),
                anatomical_validated=bool(self.validated and visible), source="local_custom_yolo_onnx",
                reason="Annotated evaluation supplied" if self.validated else "Anatomical accuracy not evaluated; excluded from Kang geometry")
        return records


class HeadbandPoleDetector:
    """Future automatic marker/keypoint geometry; no pole inferred from face shape."""
    def detect(self, image):
        return {name: dict(name=name, status="UNRESOLVED", pixel=None, confidence=None,
                           visible=False, anatomical_validated=False,
                           reason="Headband pole position has no validated geometric detector")
                for name in ("left_headband_pole", "right_headband_pole")}
