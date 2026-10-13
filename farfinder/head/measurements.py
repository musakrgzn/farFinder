"""Euclidean and explicitly approximate surface measurements with availability gates."""

import csv
from pathlib import Path

import numpy as np

from farfinder.head.geometry import vector
from farfinder.head.session import save_json


PAIRS = [
    ("glabella_to_posterior_reference", "G", "O"),
    ("left_ear_to_right_ear", "left_tragus", "right_tragus"),
    ("V_to_P", "V", "P"),
    ("V_to_left_ear_reference", "V", "left_tragus"),
    ("V_to_right_ear_reference", "V", "right_tragus"),
    ("FL_to_FR", "FL", "FR")]


def euclidean_distance(start, end):
    return float(np.linalg.norm(vector(start)-vector(end)))


def measure(name, start_name, end_name, landmarks, surface, scale, distance_type):
    record = dict(measurement_name=name, start_landmark=start_name, end_landmark=end_name,
                  distance_value=None, distance_unit="cm" if scale.valid else "reconstruction_units",
                  distance_type=distance_type, metric_scale_valid=scale.valid,
                  confidence_or_uncertainty=None, quality_status="UNAVAILABLE", reason=None)
    endpoints = []
    for endpoint in (start_name, end_name):
        item = landmarks.get(endpoint)
        if not item or item.get("status") == "UNRESOLVED" or item.get("point_3d") is None:
            reason = item.get("reason", "Missing landmark") if item else "Missing landmark"
            record["reason"] = f"{endpoint}: {reason}"
            return record
        endpoints.append(item)
    try:
        a, b = [vector(item["point_3d"]) for item in endpoints]
        if distance_type == "euclidean":
            distance = euclidean_distance(a, b)
            extra = {}
        elif distance_type == "surface_mesh_edge_approximation":
            distance, _, extra = surface.shortest_path(a, b)
        else:
            raise ValueError("Unknown distance type")
    except ValueError as error:
        record["reason"] = str(error)
        return record
    factor = scale.cm_per_unit if scale.valid else 1
    endpoint_uncertainties = [item.get("uncertainty_units") for item in endpoints]
    propagated = None
    if all(value is not None for value in endpoint_uncertainties):
        propagated = float(np.sqrt(sum(value**2 for value in endpoint_uncertainties)
                                   + ((distance * scale.relative_uncertainty)**2 if scale.valid else 0)) * factor)
    record.update(distance_value=distance*factor, quality_status="ESTIMATED", reason="Registered observed geometry",
                  confidence_or_uncertainty=dict(geometric_uncertainty=propagated,
                    uncertainty_unit=record["distance_unit"], anatomical_accuracy="NOT_EVALUATED",
                    model_confidences=[item.get("confidence") for item in endpoints],
                    note="Geometric uncertainty proxy excludes unknown anatomical/model error", **extra))
    return record


def calculate_measurements(landmarks, surface, scale, anatomical_frame=None, coverage_complete=False):
    points = dict(landmarks)
    pairs = list(PAIRS)
    for name, axis in (("head_width", 0), ("head_height", 1), ("head_length", 2)):
        start_name, end_name = name+"_min", name+"_max"
        extent = None
        if anatomical_frame and coverage_complete and surface is not None:
            origin = vector(anatomical_frame["origin"])
            basis = np.asarray(anatomical_frame["basis"], float)
            if basis.shape != (3,3) or not np.allclose(basis.T@basis,np.eye(3),atol=1e-5):
                raise ValueError("Invalid anatomical coordinate frame")
            local = (surface.vertices-origin) @ basis
            a, b = np.argmin(local[:, axis]), np.argmax(local[:, axis])
            directions = local/np.maximum(np.linalg.norm(local,axis=1,keepdims=True),1e-12)
            # Camera orbit coverage alone cannot establish both surface extrema.
            enough_geometry = all(np.count_nonzero(directions[:,axis]*sign > np.cos(np.radians(18))) >= 5
                                  for sign in (-1,1))
            extent = float(np.ptp(local[:,axis])) if enough_geometry else None
            for key, index in ((start_name, a), (end_name, b)):
                points[key] = (dict(status="ESTIMATED", point_3d=surface.vertices[index].tolist(), confidence=None,
                                   uncertainty_units=surface.spacing, reason="Observed outer hair/skin extent in anatomical frame")
                               if enough_geometry else dict(status="UNRESOLVED",point_3d=None,
                                   reason="Missing observed geometry near one or both anatomical surface extrema"))
        else:
            for key in (start_name, end_name):
                points[key] = dict(status="UNRESOLVED", point_3d=None,
                                  reason="Validated anatomical frame and complete orbit/top coverage required")
        pairs.append((name, start_name, end_name, extent))
    results = []
    for pair in pairs:
        name,start,end = pair[:3]
        record = measure(name,start,end,points,surface,scale,"euclidean")
        if len(pair)==4:
            record["distance_type"]="anatomical_axis_extent"
            if pair[3] is not None:
                record["distance_value"]=pair[3]*(scale.cm_per_unit if scale.valid else 1)
                record["reason"]="Observed outer hair/skin extent along anatomical axis; skull size and completeness not guaranteed"
        results.append(record)
        # Bounding dimensions are straight extents; a surface route is a distinct measurement.
        results.append(measure(name, start, end, points, surface, scale, "surface_mesh_edge_approximation"))
    return results


def save_measurements(session, measurements):
    save_json(session.path / "measurements.json", dict(schema_version=1, measurements=measurements,
              unit_policy="Centimeters only with independently validated metric scale",
              surface_policy="Mesh-edge shortest paths approximate continuous geodesics"))
    fields = ["measurement_name", "start_landmark", "end_landmark", "distance_value", "distance_unit",
              "distance_type", "metric_scale_valid", "confidence_or_uncertainty", "quality_status", "reason"]
    import json
    with (session.path / "measurements.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in measurements:
            record = dict(record)
            record["confidence_or_uncertainty"] = json.dumps(record["confidence_or_uncertainty"], allow_nan=False)
            writer.writerow(record)
