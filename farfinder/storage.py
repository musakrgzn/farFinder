"""Atomic calibration persistence and flushed, per-session CSV logs."""

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from uuid import uuid4

from farfinder.calibration import Calibration, CalibrationError, CaptureResult
from farfinder.measurement import Reading


class DataStore:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.calibration_path = self.directory / "calibration.json"

    def load(self, camera: str, width: int, height: int) -> Calibration | None:
        if not self.calibration_path.exists():
            return None
        try:
            payload = json.loads(self.calibration_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise CalibrationError("Calibration must be a JSON object.")
            calibration = Calibration.from_dict(payload)
            calibration.check_setup(camera, width, height)
            return calibration
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise CalibrationError(f"Cannot read calibration: {exc}") from exc

    def save(self, calibration: Calibration, captures: list[CaptureResult]) -> None:
        # JSON is authoritative. An interrupted write cannot destroy the old file.
        name = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=self.directory,
                                             prefix=".calibration-", suffix=".tmp",
                                             delete=False, encoding="utf-8") as file:
                name = file.name
                json.dump(calibration.to_dict(), file, indent=2, allow_nan=False)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(name, self.calibration_path)
        finally:
            if name is not None and os.path.exists(name):
                os.unlink(name)
        path = self.directory / "calibration.csv"
        fields = ["created_at_utc", "calibration_id", "camera", "width", "height",
                  "target_cm", "eye_span_px", "sample_count", "relative_std",
                  "scale", "offset_cm"]
        header_needed = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            if header_needed:
                writer.writeheader()
            for capture in captures:
                writer.writerow(dict(created_at_utc=calibration.created_at,
                                     calibration_id=calibration.calibration_id,
                                     camera=calibration.camera, width=calibration.width,
                                     height=calibration.height, target_cm=capture.target_cm,
                                     eye_span_px=capture.span_px,
                                     sample_count=capture.sample_count,
                                     relative_std=capture.relative_std,
                                     scale=calibration.scale, offset_cm=calibration.offset_cm))
            file.flush()
            os.fsync(file.fileno())

    def reset(self) -> None:
        self.calibration_path.unlink(missing_ok=True)


class MeasurementLogger:
    fields = ["timestamp_utc", "elapsed_s", "distance_cm", "raw_distance_cm",
              "eye_span_px", "fps", "quality", "calibrated",
              "within_calibrated_range", "yaw_deg", "pitch_deg", "roll_deg",
              "calibration_id"]

    def __init__(self, directory: Path, interval: float = 0.1):
        if interval <= 0:
            raise ValueError("Log interval must be positive.")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.path = Path(directory) / f"measurements_{stamp}_{uuid4().hex[:8]}.csv"
        self.file = self.path.open("x", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fields)
        self.writer.writeheader()
        self.file.flush()
        self.started = time.monotonic()
        self.last_write = float("-inf")
        self.last_flush = self.started
        self.interval = interval

    def write(self, now: float, reading: Reading, span_px: float | None,
              fps: float, calibration: Calibration | None,
              pose: tuple[float, float, float] | None = None) -> None:
        if now - self.last_write < self.interval:
            return
        yaw, pitch, roll = pose if pose is not None else (None, None, None)
        def number(value: float | None) -> str:
            return "" if value is None else f"{value:.4f}"
        self.writer.writerow(dict(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_s=number(now - self.started), distance_cm=number(reading.distance_cm),
            raw_distance_cm=number(reading.raw_distance_cm), eye_span_px=number(span_px),
            fps=number(fps), quality=reading.quality, calibrated=int(calibration is not None),
            within_calibrated_range=int(reading.within_calibrated_range),
            yaw_deg=number(yaw), pitch_deg=number(pitch), roll_deg=number(roll),
            calibration_id=calibration.calibration_id if calibration else ""))
        self.last_write = now
        if now - self.last_flush >= 1:
            self.file.flush()
            self.last_flush = now

    def close(self) -> None:
        self.file.flush()
        self.file.close()

    def __enter__(self) -> "MeasurementLogger":
        return self

    def __exit__(self, *args) -> None:
        self.close()
