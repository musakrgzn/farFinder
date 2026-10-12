"""Distance smoothing and rejection without reusing stale readings."""

from collections import deque
from dataclasses import dataclass
import math
from statistics import median

from farfinder.calibration import Calibration


class DistanceSmoother:
    def __init__(self, window: int = 5, alpha: float = 0.35):
        if window < 1 or not 0 < alpha <= 1:
            raise ValueError("Invalid smoothing settings.")
        self.values: deque[float] = deque(maxlen=window)
        self.alpha = alpha
        self.value: float | None = None

    def reset(self) -> None:
        self.values.clear()
        self.value = None

    def update(self, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Distance must be finite and positive.")
        self.values.append(value)
        filtered = median(self.values)
        self.value = (filtered if self.value is None
                      else self.alpha * filtered + (1 - self.alpha) * self.value)
        return self.value


@dataclass(frozen=True)
class Reading:
    distance_cm: float | None
    raw_distance_cm: float | None
    quality: str
    within_calibrated_range: bool = False


class MeasurementEngine:
    def __init__(self, calibration: Calibration | None = None):
        self.calibration = calibration
        self.smoother = DistanceSmoother()

    def set_calibration(self, calibration: Calibration | None) -> None:
        self.calibration = calibration
        self.smoother.reset()

    def process(self, span_px: float | None, quality: str = "OK") -> Reading:
        if quality != "OK" or span_px is None:
            self.smoother.reset()
            return Reading(None, None, quality if quality != "OK" else "Invalid eye span")
        if not math.isfinite(span_px) or span_px < 40:
            self.smoother.reset()
            return Reading(None, None, "Eye span too small or invalid")
        if self.calibration is None:
            self.smoother.reset()
            return Reading(None, None, "Not calibrated - press 1 at 30 cm")
        raw = self.calibration.estimate(span_px)
        if not math.isfinite(raw) or not 20 <= raw <= 120:
            self.smoother.reset()
            return Reading(None, None, "Outside estimated 20-120 cm range")
        within = 30 <= raw <= 60
        return Reading(self.smoother.update(raw), raw,
                       "OK" if within else "Extrapolated - outside 30-60 cm", within)
