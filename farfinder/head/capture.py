"""Automatic masked multi-view selection with feature overlap and blur gates."""

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np

from farfinder.head.capture_mode import capture_assumptions


VIEW_NAMES = ["Front", "Front-left", "Left", "Back-left", "Back",
              "Back-right", "Right", "Front-right", "Top"]


@dataclass
class HeadMask:
    mask: np.ndarray
    categories: np.ndarray
    confidence: np.ndarray
    reason: str = "OK"
    source: str = "hair_face_segmentation"
    tracking_frames: int = 0
    tracking_inliers: int = 0
    tracking_error_px: float | None = None


class HeadMaskTracker:
    """Bridge short semantic dropouts using head features and current foreground.

    A warped old mask alone is never accepted. Tracking starts from a detected
    head, requires forward/backward optical-flow agreement, and expires without
    a fresh hair/face anchor. It does not supply anatomical landmarks.
    """
    def __init__(self, max_frames=30, max_seconds=3.0):
        self.max_frames = max_frames
        self.max_seconds = max_seconds
        self.reset()

    def reset(self):
        self.gray = self.mask = None
        self.anchor_time = None
        self.tracking_frames = 0

    def _predict(self, gray):
        if self.gray is None or self.gray.shape != gray.shape:
            return None
        points = cv2.goodFeaturesToTrack(self.gray, maxCorners=180, qualityLevel=0.015,
                                         minDistance=5, mask=cv2.erode(self.mask,np.ones((5,5),np.uint8)),
                                         blockSize=7)
        if points is None or len(points)<24:
            return None
        settings = dict(winSize=(21,21),maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
        moved, forward_ok, errors = cv2.calcOpticalFlowPyrLK(self.gray,gray,points,None,**settings)
        if moved is None:
            return None
        backwards, backward_ok, _ = cv2.calcOpticalFlowPyrLK(gray,self.gray,moved,None,**settings)
        if backwards is None:
            return None
        fb_error = np.linalg.norm(backwards.reshape(-1,2)-points.reshape(-1,2),axis=1)
        good = (forward_ok.ravel()>0)&(backward_ok.ravel()>0)&(fb_error<1.0)&(errors.ravel()<25)
        good &= np.isfinite(moved.reshape(-1,2)).all(axis=1)
        if np.count_nonzero(good)<20:
            return None
        transform, inliers = cv2.estimateAffinePartial2D(points[good],moved[good],
            method=cv2.RANSAC,ransacReprojThreshold=2.5,maxIters=2000,confidence=0.995)
        if transform is None or inliers is None or np.count_nonzero(inliers)<16 or np.mean(inliers)<0.65:
            return None
        scale = float(np.linalg.norm(transform[:,0]))
        rotation = float(np.degrees(np.arctan2(transform[1,0],transform[0,0])))
        flow = float(np.median(np.linalg.norm(moved[good]-points[good],axis=2)))
        if not 0.8<=scale<=1.25 or abs(rotation)>25 or flow>gray.shape[1]*0.15:
            return None
        predicted = cv2.warpAffine(self.mask,transform,(gray.shape[1],gray.shape[0]),flags=cv2.INTER_NEAREST)
        return predicted, moved[good][inliers.ravel()>0].reshape(-1,2), float(np.median(fb_error[good]))

    def resolve(self, frame, semantic, probabilities, now=None):
        now = time.monotonic() if now is None else now
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        area = int(np.count_nonzero(semantic.mask))
        fraction = area/gray.size
        reliable = semantic.reason=="OK" and 0.04<fraction<0.85
        if semantic.reason=="Multiple head regions":
            self.reset()
            return semantic
        prediction = self._predict(gray) if self.mask is not None else None
        if reliable and (prediction is None or area>=np.count_nonzero(prediction[0])*0.5):
            self.gray,self.mask = gray.copy(),semantic.mask.copy()
            self.anchor_time = now
            self.tracking_frames = 0
            return semantic
        expired = (self.anchor_time is not None and
                   (now-self.anchor_time>self.max_seconds or self.tracking_frames>=self.max_frames))
        if expired:
            self.reset()
            return HeadMask(np.zeros_like(semantic.mask),semantic.categories,semantic.confidence,
                "Head tracking expired - return to a clear front view to reacquire",source="unresolved")
        if prediction is None:
            self.reset()
            return semantic
        predicted, points, error = prediction
        foreground_probability = np.clip(probabilities[[1,2,3]].sum(axis=0),0,1)
        foreground = np.isin(semantic.categories,[1,2,3])&(foreground_probability>=0.75)
        candidate = ((predicted>0)&foreground).astype(np.uint8)
        if np.count_nonzero(candidate)<np.count_nonzero(predicted)*0.65:
            self.reset()
            return HeadMask(np.zeros_like(semantic.mask),semantic.categories,semantic.confidence,
                "Head tracking lost foreground support - return toward the last clear view",source="unresolved")
        count,labels,stats,_ = cv2.connectedComponentsWithStats(candidate)
        if count<=1:
            self.reset()
            return semantic
        main = max(range(1,count),key=lambda index:stats[index,cv2.CC_STAT_AREA])
        mask = (labels==main).astype(np.uint8)*255
        pixels = np.rint(points).astype(int)
        inside = ((pixels[:,0]>=0)&(pixels[:,0]<gray.shape[1])&
                  (pixels[:,1]>=0)&(pixels[:,1]<gray.shape[0]))
        supported = np.zeros(len(points),bool)
        supported[inside] = mask[pixels[inside,1],pixels[inside,0]]>0
        if np.mean(supported)<0.75 or not 0.04<np.mean(mask>0)<0.85:
            self.reset()
            return HeadMask(np.zeros_like(mask),semantic.categories,semantic.confidence,
                            "Head tracking uncertain - reacquire a clear front view",source="unresolved")
        self.gray,self.mask = gray.copy(),mask.copy()
        self.tracking_frames+=1
        return HeadMask(mask,semantic.categories,foreground_probability,source="temporal_tracking",
                        tracking_frames=self.tracking_frames,tracking_inliers=int(np.count_nonzero(supported)),
                        tracking_error_px=error)


class HeadSegmenter:
    """Hair/face segmentation, including adjacent skin; never uses background masks."""
    def __init__(self, model_path=Path("/app/models/selfie_multiclass.tflite")):
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path),
                                              delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            output_category_mask=True, output_confidence_masks=True)
        self.segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options)
        labels = self.segmenter.labels
        if labels != ["background","hair","body-skin","face-skin","clothes","others"]:
            self.segmenter.close()
            raise RuntimeError(f"Unexpected multiclass model labels: {labels}")
        self.tracker = HeadMaskTracker()

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        categories = result.category_mask.numpy_view().copy().squeeze()
        probabilities = np.stack([m.numpy_view().squeeze() for m in result.confidence_masks])
        semantic = self.semantic_mask(categories,probabilities)
        return self.tracker.resolve(frame,semantic,probabilities)

    @staticmethod
    def semantic_mask(categories, probabilities):
        # Model classes: background, hair, body-skin, face-skin, clothes, others.
        base = ((categories == 1) | (categories == 3)) & (probabilities[[1, 3]].max(axis=0) >= 0.65)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(base.astype(np.uint8))
        mask = np.zeros(base.shape, dtype=np.uint8)
        if count <= 1:
            return HeadMask(mask, categories, probabilities.max(axis=0),
                            "No reliable hair/face head mask (bald rear views may be unsupported)")
        ordered = sorted(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
        main = ordered[0]
        if (len(ordered) > 1 and stats[ordered[1], cv2.CC_STAT_AREA] >
                max(300, stats[main, cv2.CC_STAT_AREA] * 0.35)):
            return HeadMask(mask, categories, probabilities.max(axis=0), "Multiple head regions")
        x, y, w, h, area = stats[main]
        pad = max(3, round(w * 0.12))
        roi = np.zeros_like(base)
        roi[max(0, y-pad):min(base.shape[0], y+h+3),
            max(0, x-pad):min(base.shape[1], x+w+pad)] = True
        skin = (categories == 2) & (probabilities[2] >= 0.7) & roi
        candidate = ((labels == main) | skin).astype(np.uint8)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        count, labels2, stats2, _ = cv2.connectedComponentsWithStats(candidate)
        main2 = max(range(1, count), key=lambda i: stats2[i, cv2.CC_STAT_AREA])
        mask[labels2 == main2] = 255
        return HeadMask(mask, categories, probabilities[[1, 2, 3]].max(axis=0))

    def close(self):
        self.segmenter.close()

    def reset_tracking(self):
        self.tracker.reset()


def orientation_label(pose):
    if pose is None:
        return "UNRESOLVED"
    yaw, pitch, roll = pose
    if abs(pitch) > 45:
        return "Top (estimated)"
    return VIEW_NAMES[int(round((yaw % 360) / 45)) % 8] + " (estimated)"


class CaptureSelector:
    def __init__(self, min_interval=0.2, capture_mode="moving-camera"):
        capture_assumptions(capture_mode)
        self.capture_mode = capture_mode
        self.sift = cv2.SIFT_create(nfeatures=1600)
        self.previous = None
        self.last_save = -float("inf")
        self.min_interval = min_interval

    def evaluate(self, image, head: HeadMask, now, pose=None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = head.mask
        metadata = dict(sharpness=None, mask_coverage=float(np.mean(mask > 0)),
                        overlap=None, flow_px=None, feature_count=0, status="REJECTED",
                        view=orientation_label(pose))
        metadata.update(head_mask_source=head.source,head_mask_status="ESTIMATED",
                        tracking_frames=head.tracking_frames,tracking_inliers=head.tracking_inliers,
                        tracking_error_px=head.tracking_error_px,capture_mode=self.capture_mode)
        def reject(reason):
            metadata["reason"] = reason
            return False, metadata
        if head.reason != "OK":
            return reject(head.reason)
        if self.capture_mode=="head-rotation" and self.previous is None:
            if pose is None or not np.isfinite(pose).all() or abs(pose[0])>15 or abs(pose[1])>20:
                return reject("Start head-turn capture facing the webcam directly, then turn slowly toward both sides")
        ys, xs = np.where(mask > 0)
        if not 0.04 < metadata["mask_coverage"] < 0.85 or len(xs) == 0:
            return reject("Head coverage must be between 4% and 85%")
        height, width = gray.shape
        if min(xs) < 2 or max(xs) >= width-2 or min(ys) < 2 or max(ys) >= height-2:
            return reject("Head clipped by image border")
        brightness = float(gray[mask > 0].mean())
        metadata["brightness"] = brightness
        eroded = cv2.erode(mask, np.ones((5, 5), np.uint8))
        if not np.any(eroded):
            return reject("Head region too small")
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F)[eroded > 0].var())
        metadata["sharpness"] = sharpness
        if sharpness < 50 or not 35 < brightness < 240:
            return reject("Head blurred or poorly lit")
        keypoints, descriptors = self.sift.detectAndCompute(gray, eroded)
        metadata["feature_count"] = len(keypoints)
        if descriptors is None or len(keypoints) < 70:
            return reject("Insufficient head texture for reconstruction")
        if now - self.last_save < self.min_interval:
            return reject("Waiting for capture interval")
        if self.previous:
            previous_points, previous_descriptors = self.previous
            pairs = cv2.BFMatcher().knnMatch(previous_descriptors, descriptors, k=2)
            matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
            if len(matches) < 25:
                return reject("Insufficient overlap; turn back toward the last saved head angle"
                              if self.capture_mode=="head-rotation"
                              else "Insufficient overlap; move camera back toward the last view")
            a = np.float32([previous_points[m.queryIdx].pt for m in matches])
            b = np.float32([keypoints[m.trainIdx].pt for m in matches])
            _, inliers = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 1.5, 0.995)
            if inliers is None or np.count_nonzero(inliers) < 20:
                return reject("No geometrically consistent head overlap")
            valid = inliers.ravel().astype(bool)
            metadata["geometric_inlier_fraction"] = float(np.mean(valid))
            if self.capture_mode=="head-rotation" and np.mean(valid)<0.65:
                return reject("Head features inconsistent; keep expression/hair stable and use smaller turns")
            overlap = float(np.count_nonzero(valid) / min(len(previous_points), len(keypoints)))
            flow = float(np.median(np.linalg.norm(a[valid] - b[valid], axis=1)))
            metadata.update(overlap=overlap, flow_px=flow)
            if overlap < 0.08:
                return reject("Overlap too low")
            if flow < max(3, width * 0.008):
                return reject("Redundant frame; turn head slowly" if self.capture_mode=="head-rotation"
                              else "Redundant frame; move camera, keep subject still")
            if flow > width * 0.22:
                return reject("Head turn too large; return toward the last saved view" if self.capture_mode=="head-rotation"
                              else "Camera step too large; capture overlapping intermediate views")
        self.previous = keypoints, descriptors
        self.last_save = now
        metadata.update(status="ACCEPTED", reason="Sharp, textured, overlapping head frame")
        return True, metadata
