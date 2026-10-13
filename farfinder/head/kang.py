"""Conservative Kang reference geometry following the user-supplied definitions.

No generic facial point substitutes for a tragus, ear attachment, or headband pole.
Geometry is gated on validated references and an observed scalp surface.
"""

import numpy as np

from farfinder.head.geometry import plane_pair_intersections, vector


def unresolved(reason):
    return dict(status="UNRESOLVED", point_3d=None, confidence=None, visible=False,
                uncertainty_units=None, anatomical_validated=False, reason=reason)


def require(landmarks, names, anatomical=False):
    for name in names:
        item = landmarks.get(name)
        if not item or item.get("status") == "UNRESOLVED" or item.get("point_3d") is None:
            raise ValueError(f"Required reference {name} is unresolved")
        vector(item["point_3d"])
        if anatomical and not item.get("anatomical_validated", False):
            raise ValueError(f"{name} lacks validated anatomical detection")
        if item.get("uncertainty_units") is None:
            raise ValueError(f"{name} has no reconstruction uncertainty estimate")
        if not np.isfinite(item["uncertainty_units"]) or item["uncertainty_units"]<0:
            raise ValueError(f"{name} has invalid reconstruction uncertainty")


def anatomical_frame(landmarks):
    require(landmarks, ["left_tragus", "right_tragus"], anatomical=True)
    require(landmarks, ["left_eyebrow_peak", "right_eyebrow_peak", "chin", "nose_tip"])
    left = vector(landmarks["left_tragus"]["point_3d"])
    right = vector(landmarks["right_tragus"]["point_3d"])
    origin = (left+right)/2
    width = np.linalg.norm(left-right)
    if width <= 1e-8:
        raise ValueError("Degenerate bilateral tragus references")
    for name in ("left_tragus", "right_tragus"):
        if landmarks[name]["uncertainty_units"] > width*0.1:
            raise ValueError("Bilateral tragus reconstruction uncertainty too large")
    x = (left-right)/width  # subject's left
    brows = (vector(landmarks["left_eyebrow_peak"]["point_3d"])+
             vector(landmarks["right_eyebrow_peak"]["point_3d"]))/2
    up = brows-vector(landmarks["chin"]["point_3d"])
    up -= x*np.dot(up, x)
    if np.linalg.norm(up) < width*0.1:
        raise ValueError("Facial superior direction is degenerate")
    y = up/np.linalg.norm(up)
    z = np.cross(x, y)  # anterior; signs checked against the registered nose
    if np.dot(vector(landmarks["nose_tip"]["point_3d"])-origin, z) <= width*0.05:
        raise ValueError("Bilateral labels and frontal orientation are inconsistent")
    return dict(origin=origin.tolist(), basis=np.column_stack([x, y, z]).tolist(),
                axes=["subject_left", "superior", "anterior"], status="ESTIMATED",
                reason="Validated bilateral ears; facial landmarks supply orientation, not Kang references")


def calculate_kang(surface, landmarks, scalp_surface_validated=False, reconstruction_uncertainty=0,
                   hairline=None):
    results = {name: unresolved("Required anatomy and observed scalp surface must be validated")
               for name in ("V", "P", "FL", "FR")}
    frame = None
    try:
        frame = anatomical_frame(landmarks)
    except ValueError as error:
        return {name: unresolved(str(error)) for name in results}, frame
    if not scalp_surface_validated:
        return {name: unresolved("Observed outer hair/skin mesh has not been validated as the required scalp surface")
                for name in results}, frame
    origin, basis = vector(frame["origin"]), np.asarray(frame["basis"])
    x, y, z = basis.T
    def resolved(point, names, reason):
        uncertainty = float(np.sqrt(surface.spacing**2 + reconstruction_uncertainty**2 +
                          sum(landmarks[name]["uncertainty_units"]**2 for name in names)))
        return dict(status="ESTIMATED", point_3d=point.tolist(), visible=True,
                    confidence=min((landmarks[name].get("confidence") or 0) for name in names),
                    uncertainty_units=uncertainty, anatomical_validated=False, reason=reason,
                    accuracy="Geometric estimate; Kang anatomical accuracy not evaluated")
    try:
        hits = plane_pair_intersections(surface, x, np.dot(x, origin), z, np.dot(z, origin))
        superior = [point for point in hits if np.dot(point-origin, y) > surface.spacing]
        if not superior:
            raise ValueError("No observed superior scalp intersection of midsagittal/coronal references")
        point = max(superior, key=lambda p: np.dot(p-origin, y))
        results["V"] = resolved(point, ["left_tragus", "right_tragus"],
                                "Superior observed intersection of midsagittal and bilateral-tragus coronal planes")
    except ValueError as error:
        results["V"] = unresolved(str(error))
    try:
        roots = ["left_superior_ear_root", "right_superior_ear_root"]
        require(landmarks, roots, anatomical=True)
        a, b = [vector(landmarks[name]["point_3d"]) for name in roots]
        if np.dot(a-origin, x) <= 0 or np.dot(b-origin, x) >= 0:
            raise ValueError("Superior ear-root left/right labels are inconsistent")
        level_uncertainty = np.hypot(landmarks[roots[0]]["uncertainty_units"],landmarks[roots[1]]["uncertainty_units"])
        if abs(np.dot(a-b,y)) > max(level_uncertainty*3,surface.spacing*0.25):
            raise ValueError("Upper ear attachments do not establish a consistent horizontal level within uncertainty")
        height = np.dot((a+b)/2, y)
        if height <= np.dot(origin, y):
            raise ValueError("Superior ear attachments must be above the tragus reference")
        hits = plane_pair_intersections(surface, x, np.dot(x, origin), y, height)
        posterior = [point for point in hits if np.dot(point-origin, z) < -surface.spacing]
        if not posterior:
            raise ValueError("No observed posterior midsagittal scalp intersection at ear-root height")
        point = min(posterior, key=lambda p: np.dot(p-origin, z))
        results["P"] = resolved(point, roots+["left_tragus", "right_tragus"],
                                "Posterior observed intersection at bilateral upper-ear-attachment horizontal level")
    except ValueError as error:
        results["P"] = unresolved(str(error))
    for name, side in (("FL", "left"), ("FR", "right")):
        try:
            pole, brow = f"{side}_headband_pole", f"{side}_eyebrow_peak"
            require(landmarks, [pole], anatomical=True)
            require(landmarks, [brow])
            if not hairline or hairline.get("status") == "UNRESOLVED" or not hairline.get("points_3d"):
                raise ValueError("Frontal hairline has no multi-view surface registration")
            _, path, path_quality = surface.shortest_path(landmarks[pole]["point_3d"], landmarks[brow]["point_3d"])
            candidates = np.asarray(hairline["points_3d"], float)
            side_sign = 1 if side == "left" else -1
            candidates = candidates[((candidates-origin) @ x)*side_sign > 0]
            if not len(candidates):
                raise ValueError(f"No registered {side} frontal hairline")
            # Locate an intersection with the observed surface tape-path approximation.
            distances = np.linalg.norm(candidates[:, None]-path[None, :], axis=2)
            accepted = np.flatnonzero(distances.min(axis=1) <= surface.spacing*1.5)
            if not len(accepted):
                raise ValueError("Hairline does not intersect the observed pole-to-brow tape path")
            path_indices = distances[accepted].argmin(axis=1)
            if np.ptp(path_indices) > max(3, len(path)*0.15):
                raise ValueError("Ambiguous multiple frontal hairline/tape-path intersections")
            point = candidates[accepted[np.argmin(distances[accepted].min(axis=1))]]
            results[name] = resolved(point, [pole, brow],
                                     "Registered frontal hairline intersecting mesh-edge tape-path approximation")
            results[name]["tape_path"] = path_quality
        except ValueError as error:
            results[name] = unresolved(str(error))
    return results, frame
