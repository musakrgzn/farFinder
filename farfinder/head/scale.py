"""SfM scale remains arbitrary until an independent physical reference validates it."""

from dataclasses import asdict, dataclass

import numpy as np

from farfinder.head.geometry import triangulate
from farfinder.head.intrinsics import BoardSpec


@dataclass(frozen=True)
class MetricScale:
    valid: bool = False
    cm_per_unit: float | None = None
    relative_uncertainty: float | None = None
    source: str = "UNRESOLVED"
    reason: str = "Monocular 30/60 cm face calibration does not fix SfM scale"
    reference_count: int = 0

    def __post_init__(self):
        if type(self.valid) is not bool:
            raise ValueError("Metric validity must be an explicit boolean")
        if self.valid and (self.cm_per_unit is None or not np.isfinite(self.cm_per_unit)
                           or self.cm_per_unit <= 0 or self.relative_uncertainty is None
                           or not np.isfinite(self.relative_uncertainty)
                           or not 0 <= self.relative_uncertainty <= 0.1
                           or self.reference_count < 1
                           or self.source not in ("independently_measured_dimension","charuco_independently_measured_square")):
            raise ValueError("Invalid metric-scale certificate")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__ if key in payload})


def validate_reference(reconstruction_length, measured_cm, uncertainty_cm,
                       independently_measured=False, reprojection_rms_px=0, views=3):
    if not independently_measured:
        return MetricScale(reason="Reference dimension has not been independently measured")
    if (not all(np.isfinite(v) and v > 0 for v in (reconstruction_length, measured_cm, uncertainty_cm))
            or uncertainty_cm/measured_cm > 0.05 or views < 3
            or not np.isfinite(reprojection_rms_px) or reprojection_rms_px > 3):
        return MetricScale(reason="Reference uncertainty, geometry, or camera registration insufficient")
    return MetricScale(True, measured_cm/reconstruction_length, uncertainty_cm/measured_cm,
                       "independently_measured_dimension", "Validated independent reference", 1)


def scale_from_board(session, cameras, measured_square_cm=None, uncertainty_cm=None):
    if session.metadata.get("capture_mode", "moving-camera")=="head-rotation":
        return MetricScale(reason="Head-turn poses are head-relative. A stationary board does not move with the head; "
                                  "this mode has no validated co-moving physical scale reference")
    if measured_square_cm is None or uncertainty_cm is None:
        return MetricScale(reason="Supply independently measured printed square size and its uncertainty")
    if (not np.isfinite(measured_square_cm) or measured_square_cm <= 0
            or not np.isfinite(uncertainty_cm) or uncertainty_cm <= 0
            or uncertainty_cm/measured_square_cm > 0.05):
        return MetricScale(reason="Invalid measured board dimension/uncertainty")
    intrinsic = session.metadata.get("intrinsics")
    if not intrinsic:
        return MetricScale(reason="No intrinsic calibration for board registration")
    spec = BoardSpec(**(intrinsic["board"] | {"square_cm": measured_square_cm}))
    reference_points = spec.board().getChessboardCorners()
    tracks = {}
    for frame in session.metadata["frames"]:
        board = frame.get("board_observation")
        if frame["image"] not in cameras or not board:
            continue
        for index, pixel in zip(board["ids"], board["pixels"]):
            tracks.setdefault(index, []).append(dict(image=frame["image"], pixel=pixel))
    recovered, known, uncertainties = [], [], []
    for index, observations in tracks.items():
        if not 0 <= index < len(reference_points):
            continue
        try:
            point, quality = triangulate(observations, cameras, min_views=3)
        except ValueError:
            continue
        recovered.append(point)
        known.append(reference_points[index])
        uncertainties.append(quality["uncertainty_units"])
    if len(recovered) < 8:
        return MetricScale(reason="Need at least 8 board corners triangulated in 3 registered views")
    a, b = np.asarray(recovered), np.asarray(known)
    a_center, b_center = a.mean(axis=0), b.mean(axis=0)
    x, y = a-a_center, b-b_center
    u, singular, vt = np.linalg.svd(y.T @ x / len(x))
    sign = np.eye(3)
    sign[2, 2] = np.linalg.det(u @ vt)
    rotation = u @ sign @ vt
    scale = float(np.sum(singular * np.diag(sign)) / np.mean(np.sum(x*x, axis=1)))
    residuals = np.linalg.norm(scale * x @ rotation.T-y, axis=1)
    reference_extent = float(np.linalg.norm(np.ptp(b, axis=0)))
    residual_fraction = float(np.sqrt(np.mean(residuals**2)) / reference_extent)
    geometric_fraction = float(np.median(uncertainties) / np.linalg.norm(np.ptp(a, axis=0)))
    relative = float(np.sqrt((uncertainty_cm/measured_square_cm)**2 + residual_fraction**2 + geometric_fraction**2))
    if (np.linalg.matrix_rank(y, tol=1e-5) < 2 or scale <= 0 or residual_fraction > 0.025
            or relative > 0.1):
        return MetricScale(reason="Board fails rigid similarity/uncertainty checks; board or person may have moved")
    return MetricScale(True, scale, relative, "charuco_independently_measured_square",
                       "Static board, triangulated corners, measured square size, rigid-fit validation", len(a))
