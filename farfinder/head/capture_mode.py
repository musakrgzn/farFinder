"""Capture assumptions and conservative gates for fixed-webcam head turns."""

import math


CAPTURE_MODES = ("moving-camera", "head-rotation")


def capture_assumptions(mode="moving-camera"):
    if mode not in CAPTURE_MODES:
        raise ValueError(f"Unsupported capture mode: {mode}")
    if mode == "head-rotation":
        return dict(capture_mode=mode, experimental=True,
                    pose_coordinate_frame="head_relative",
                    stationarity="Fixed webcam; approximately rigid head rotation. Not independently verified.",
                    instructions="Fixed webcam: start frontal, turn slowly toward each side, keep a neutral expression.")
    return dict(capture_mode=mode, experimental=False,
                pose_coordinate_frame="stationary_scene",
                stationarity="Stationary subject; moving camera. Not independently verified.",
                instructions="Keep subject stationary. Move camera slowly around head and above it.")


def head_turn_coverage(frames):
    yaws = []
    for frame in frames:
        pose = frame.get("estimated_head_orientation") or {}
        yaw = pose.get("yaw_deg")
        if isinstance(yaw, (int, float)) and math.isfinite(yaw):
            yaws.append(float(yaw))
    low, high = (min(yaws), max(yaws)) if yaws else (None, None)
    return dict(status="ESTIMATED", oriented_frames=len(yaws), min_yaw_deg=low,
                max_yaw_deg=high, yaw_span_deg=high-low if yaws else 0,
                frontal_anchor=any(abs(yaw)<=15 for yaw in yaws),
                both_sides=bool(yaws and low<=-15 and high>=15),
                source="Facial pose estimates; not a rigidity or reconstruction certificate")


def validate_head_turns(frames):
    coverage = head_turn_coverage(frames)
    if len(frames)<12:
        raise ValueError("Head-turn mode needs at least 12 overlapping frames; aim for 30-60 covering both sides")
    if (coverage["oriented_frames"]<6 or not coverage["frontal_anchor"] or
            not coverage["both_sides"] or coverage["yaw_span_deg"]<40):
        raise ValueError(f"Head-turn views too similar ({coverage['yaw_span_deg']:.0f} degree saved yaw span). "
                         "Start frontal, then turn slowly toward both sides; aim for at least 40 degrees total")
    return coverage
