"""ChArUco intrinsics are separate from face-size distance calibration."""

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import cv2
import numpy as np

from farfinder.head.session import save_json, utc_now


@dataclass(frozen=True)
class BoardSpec:
    squares_x: int = 7
    squares_y: int = 5
    square_cm: float = 2.0
    marker_ratio: float = 0.7
    dictionary: str = "DICT_5X5_100"

    def board(self):
        if (self.squares_x < 3 or self.squares_y < 3 or not np.isfinite(self.square_cm)
                or self.square_cm <= 0 or not 0.3 < self.marker_ratio < 0.9
                or self.dictionary != "DICT_5X5_100"):
            raise ValueError("Invalid ChArUco board specification")
        return cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), self.square_cm,
                                     self.square_cm * self.marker_ratio,
                                     cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    camera_matrix: list
    distortion: list
    rms_px: float
    per_view_errors_px: list
    board: dict
    created_at: str
    camera: str = "/dev/video0"
    schema_version: int = 1

    def __post_init__(self):
        k = np.asarray(self.camera_matrix, dtype=float)
        d = np.asarray(self.distortion, dtype=float)
        if (self.schema_version != 1 or type(self.width) is not int or self.width <= 0
                or type(self.height) is not int or self.height <= 0 or k.shape != (3, 3)
                or not np.isfinite(k).all() or not np.isfinite(d).all()
                or d.ndim != 1 or len(d) != 5 or not np.allclose(k[2], [0, 0, 1])
                or k[0, 0] <= 0 or k[1, 1] <= 0 or abs(k[0, 1]) > 1e-9
                or not 0 <= k[0, 2] < self.width or not 0 <= k[1, 2] < self.height
                or not np.isfinite(self.rms_px) or not 0 <= self.rms_px <= 1.5):
            raise ValueError("Invalid or insufficiently accurate camera intrinsics")
        BoardSpec(**self.board).board()

    @property
    def K(self):
        return np.asarray(self.camera_matrix, dtype=float)

    @property
    def D(self):
        return np.asarray(self.distortion, dtype=float)

    def check_resolution(self, width, height, camera=None):
        if (width, height) != (self.width, self.height) or (camera is not None and camera != self.camera):
            raise ValueError("Intrinsic calibration camera/resolution mismatch. Recalibrate; do not resize K implicitly.")

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        save_json(Path(path), self.to_dict())

    @classmethod
    def load(cls, path):
        try:
            return cls(**json.loads(Path(path).read_text()))
        except (TypeError,KeyError) as error:
            raise ValueError(f"Invalid intrinsic calibration file: {error}") from error


class BoardDetector:
    def __init__(self, spec=BoardSpec(), intrinsics=None):
        self.spec = spec
        self.board = spec.board()
        parameters = cv2.aruco.CharucoParameters()
        if intrinsics:
            parameters.cameraMatrix = intrinsics.K
            parameters.distCoeffs = intrinsics.D
        self.detector = cv2.aruco.CharucoDetector(self.board, parameters)

    def detect(self, image):
        corners, ids, _, _ = self.detector.detectBoard(image)
        if corners is None or ids is None or len(ids) < 6:
            return None
        return dict(ids=ids.reshape(-1).astype(int).tolist(),
                    pixels=corners.reshape(-1, 2).astype(float).tolist())


class IntrinsicCollector:
    def __init__(self, spec=BoardSpec(), required_views=20):
        self.spec = spec
        self.detector = BoardDetector(spec)
        self.required_views = required_views
        self.views = []
        self.signatures = []
        self.resolution = None

    def add(self, image):
        height, width = image.shape[:2]
        if self.resolution and self.resolution != (width, height):
            raise ValueError("Resolution changed during intrinsic calibration")
        self.resolution = width, height
        observation = self.detector.detect(image)
        if not observation or len(observation["ids"]) < 12:
            return False, "Show at least 12 ChArUco corners"
        points = np.array(observation["pixels"])
        box = cv2.boundingRect(points.astype(np.float32))
        x, y, w, h = box
        roi = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        if roi.size == 0 or cv2.Laplacian(roi, cv2.CV_64F).var() < 70:
            return False, "Board blurred; hold still"
        # Perspective homography signatures enforce different tilt/scale/position.
        objects = self.spec.board().getChessboardCorners()[observation["ids"], :2]
        homography, _ = cv2.findHomography(objects, points)
        if homography is None:
            return False, "Degenerate board geometry"
        signature = np.array([points[:, 0].mean()/width, points[:, 1].mean()/height,
                              w/width, h/height, homography[2, 0] * self.spec.square_cm,
                              homography[2, 1] * self.spec.square_cm])
        if any(np.linalg.norm(signature - other) < 0.09 for other in self.signatures):
            return False, "Redundant board view; vary tilt, size, and position"
        self.views.append(observation)
        self.signatures.append(signature)
        return True, f"Accepted intrinsic view {len(self.views)}/{self.required_views}"

    def calibrate(self, camera="/dev/video0"):
        if len(self.views) < self.required_views:
            raise ValueError(f"Need {self.required_views} diverse board views")
        signatures = np.asarray(self.signatures)
        if np.ptp(signatures[:, 0]) < 0.25 or np.ptp(signatures[:, 1]) < 0.2:
            raise ValueError("Move the board across more of the image before calibrating")
        objects = self.spec.board().getChessboardCorners()
        object_points = [objects[v["ids"]].astype(np.float32) for v in self.views]
        image_points = [np.asarray(v["pixels"], np.float32) for v in self.views]
        rms, k, d, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points,
                                                    self.resolution, None, None)
        errors = []
        for obj, img, rotation, translation in zip(object_points, image_points, rvecs, tvecs):
            projected, _ = cv2.projectPoints(obj, rotation, translation, k, d)
            errors.append(float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2)-img)**2, axis=1)))))
        if max(errors) > 2.0 or rms > 1.5:
            raise ValueError("Intrinsic reprojection error too high; recollect sharp diverse views")
        width, height = self.resolution
        if not 0.3 * width < k[0, 0] < 5 * width or not 0.3 * height < k[1, 1] < 8 * height:
            raise ValueError("Implausible focal lengths; recollect calibration")
        return Intrinsics(width, height, k.tolist(), d.reshape(-1).tolist(), float(rms), errors,
                          asdict(self.spec), utc_now(), camera)


def generate_board(directory, spec=BoardSpec()):
    """PNG plus true-size PDF. Printing dimensions still must be checked with a ruler."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    image = spec.board().generateImage((spec.squares_x * 160, spec.squares_y * 160), marginSize=0)
    cv2.imwrite(str(directory / "charuco.png"), image)
    margin_cm = 1.0
    figure = plt.figure(figsize=((spec.squares_x * spec.square_cm + 2*margin_cm)/2.54,
                                (spec.squares_y * spec.square_cm + 2*margin_cm)/2.54))
    axis = figure.add_axes([margin_cm/(spec.squares_x*spec.square_cm+2*margin_cm),
                           margin_cm/(spec.squares_y*spec.square_cm+2*margin_cm),
                           spec.squares_x*spec.square_cm/(spec.squares_x*spec.square_cm+2*margin_cm),
                           spec.squares_y*spec.square_cm/(spec.squares_y*spec.square_cm+2*margin_cm)])
    axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axis.axis("off")
    figure.savefig(directory / "charuco.pdf")
    plt.close(figure)
    save_json(directory / "board.json", asdict(spec))
