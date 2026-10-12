"""Calibration math and stable sample collection, independent of camera/UI."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from statistics import median, pstdev
from uuid import uuid4


class CalibrationError(ValueError):
    """Calibration is inconsistent, corrupt, or belongs to another setup."""


@dataclass(frozen=True)
class Calibration:
    scale: float
    offset_cm: float
    span_30_px: float
    span_60_px: float
    camera: str
    width: int
    height: int
    calibration_id: str
    created_at: str
    schema_version: int = 1
    feature: str = "outer_eye_corners_33_263"

    @classmethod
    def fit(cls, span_30: float, span_60: float, camera: str,
            width: int, height: int) -> "Calibration":
        scale, offset = _fit_coefficients(span_30, span_60)
        if width <= 0 or height <= 0 or not camera:
            raise CalibrationError("Camera and actual frame size are required.")
        return cls(scale, offset, span_30, span_60, camera, width, height,
                   uuid4().hex, datetime.now(timezone.utc).isoformat())

    def estimate(self, span_px: float) -> float:
        if not math.isfinite(span_px) or span_px <= 0:
            raise CalibrationError("Eye span must be finite and positive.")
        return self.scale / span_px + self.offset_cm

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "Calibration":
        try:
            calibration = cls(**payload)
            if (type(calibration.schema_version) is not int
                    or calibration.schema_version != 1
                    or calibration.feature != "outer_eye_corners_33_263"):
                raise CalibrationError("Unsupported calibration schema or feature.")
            scale, offset = _fit_coefficients(calibration.span_30_px,
                                              calibration.span_60_px)
            if not (math.isfinite(calibration.scale)
                    and math.isfinite(calibration.offset_cm)
                    and math.isclose(calibration.scale, scale, rel_tol=1e-9)
                    and math.isclose(calibration.offset_cm, offset, abs_tol=1e-9)):
                raise CalibrationError("Calibration coefficients are inconsistent.")
            if (type(calibration.width) is not int or calibration.width <= 0
                    or type(calibration.height) is not int or calibration.height <= 0
                    or not isinstance(calibration.camera, str) or not calibration.camera
                    or not isinstance(calibration.calibration_id, str)
                    or not calibration.calibration_id
                    or not isinstance(calibration.created_at, str)):
                raise CalibrationError("Invalid calibration metadata.")
            datetime.fromisoformat(calibration.created_at)
            return calibration
        except (TypeError, ValueError, OverflowError) as exc:
            raise CalibrationError(f"Invalid calibration file: {exc}") from exc

    def check_setup(self, camera: str, width: int, height: int) -> None:
        if (camera, width, height) != (self.camera, self.width, self.height):
            raise CalibrationError("Camera or frame size changed; recalibrate with 1 and 2.")


def _fit_coefficients(span_30: float, span_60: float) -> tuple[float, float]:
    if not all(math.isfinite(s) and s >= 40 for s in (span_30, span_60)):
        raise CalibrationError("Calibration needs finite eye spans of at least 40 pixels.")
    ratio = span_30 / span_60
    if not 1.4 <= ratio <= 2.8:
        raise CalibrationError("30 cm must have a clearly larger eye span than 60 cm. Retry both points.")
    scale = 30.0 / (1.0 / span_60 - 1.0 / span_30)
    offset = 30.0 - scale / span_30
    if not math.isfinite(scale) or scale <= 0 or abs(offset) > 20:
        raise CalibrationError("Implausible calibration offset; check the measured distances.")
    return scale, offset


@dataclass(frozen=True)
class CaptureResult:
    target_cm: int
    span_px: float
    sample_count: int
    relative_std: float


class StableCapture:
    """Require a contiguous, stable batch; rejected frames clear the batch."""

    def __init__(self, target_cm: int, started_at: float, sample_count: int = 45,
                 min_seconds: float = 1.5, timeout_seconds: float = 15.0,
                 max_relative_std: float = 0.03):
        if target_cm not in (30, 60):
            raise CalibrationError("Calibration targets are 30 and 60 cm.")
        if (sample_count < 3 or min_seconds <= 0 or timeout_seconds <= min_seconds
                or not 0 < max_relative_std < 1):
            raise ValueError("Invalid calibration capture settings.")
        self.target_cm = target_cm
        self.started_at = started_at
        self.sample_count = sample_count
        self.min_seconds = min_seconds
        self.timeout_seconds = timeout_seconds
        self.max_relative_std = max_relative_std
        self.samples: list[tuple[float, float]] = []

    def add(self, span_px: float | None, now: float) -> CaptureResult | None:
        if now - self.started_at >= self.timeout_seconds:
            raise CalibrationError("Capture timed out. Face the camera, hold still, then retry.")
        if span_px is None or not math.isfinite(span_px) or span_px < 40:
            self.samples.clear()
            return None
        # A move during capture starts a new stable batch immediately.
        if self.samples and abs(span_px / median(s for _, s in self.samples) - 1) > 0.08:
            self.samples.clear()
        self.samples.append((now, span_px))
        # Bound memory while keeping enough time history at high frame rates.
        while (len(self.samples) > self.sample_count
               and now - self.samples[1][0] >= self.min_seconds):
            self.samples.pop(0)
        if (len(self.samples) < self.sample_count
                or now - self.samples[0][0] < self.min_seconds):
            return None
        spans = [s for _, s in self.samples]
        center = median(spans)
        relative_std = pstdev(spans) / center
        if relative_std > self.max_relative_std:
            self.samples.clear()
            return None
        return CaptureResult(self.target_cm, center, len(spans), relative_std)

    @property
    def progress(self) -> str:
        return f"Capturing {self.target_cm} cm: {len(self.samples)}/{self.sample_count} - hold still"
