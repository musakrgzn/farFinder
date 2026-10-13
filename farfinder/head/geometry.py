"""Validated camera transforms, multi-view registration, and mesh-edge geodesics."""

from dataclasses import dataclass
import heapq
import math

import cv2
import numpy as np


def vector(value, size=3):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError("Expected a finite coordinate vector")
    return result


@dataclass
class CameraPose:
    image: str
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    distortion: np.ndarray
    width: int
    height: int

    def __post_init__(self):
        self.K = np.asarray(self.K, float)
        self.R = np.asarray(self.R, float)
        self.t = vector(self.t)
        self.distortion = np.asarray(self.distortion, float)
        if (self.K.shape != (3, 3) or self.R.shape != (3, 3)
                or not np.isfinite(self.K).all() or not np.isfinite(self.R).all()
                or not np.allclose(self.R @ self.R.T, np.eye(3), atol=1e-5)
                or not math.isclose(np.linalg.det(self.R), 1.0, abs_tol=1e-5)
                or self.K[0, 0] <= 0 or self.K[1, 1] <= 0
                or self.width <= 0 or self.height <= 0 or self.distortion.shape != (5,)
                or not np.isfinite(self.distortion).all()):
            raise ValueError("Invalid calibrated camera pose")

    @property
    def center(self):
        return -self.R.T @ self.t

    def to_camera(self, points):
        return np.asarray(points) @ self.R.T + self.t

    def project(self, points):
        points = np.asarray(points, float).reshape(-1, 3)
        rotation, _ = cv2.Rodrigues(self.R)
        projected, _ = cv2.projectPoints(points, rotation, self.t, self.K, self.distortion)
        return projected.reshape(-1, 2), self.to_camera(points)[:, 2]

    def ray(self, pixel):
        pixel = vector(pixel, 2)
        normalized = cv2.undistortPoints(pixel.reshape(1, 1, 2), self.K, self.distortion).reshape(2)
        ray = self.R.T @ np.r_[normalized, 1]
        return self.center, ray / np.linalg.norm(ray)

    def to_dict(self):
        return dict(image=self.image, K=self.K.tolist(), R_world_to_camera=self.R.tolist(),
                    t_world_to_camera=self.t.tolist(), center=self.center.tolist(),
                    distortion=self.distortion.tolist(), width=self.width, height=self.height,
                    convention="X_camera = R_world_to_camera @ X_world + t_world_to_camera",
                    geometry_unit="reconstruction_units", status="REGISTERED")

    @classmethod
    def from_dict(cls, record):
        return cls(record["image"], record["K"], record["R_world_to_camera"],
                   record["t_world_to_camera"], record["distortion"], record["width"], record["height"])


def camera_relative_to_object(camera, object_rotation, object_translation):
    """Compose known rigid object motion with a physical camera pose.

    Used for synthetic validation. Facial pose estimates are not substituted
    for registered camera extrinsics by the reconstruction pipeline.
    """
    rotation = np.asarray(object_rotation, float)
    translation = vector(object_translation)
    return CameraPose(camera.image, camera.K.copy(), camera.R @ rotation,
                      camera.R @ translation + camera.t, camera.distortion.copy(),
                      camera.width, camera.height)


def triangulate(observations, cameras, min_views=2, max_error_px=3, min_angle_deg=3):
    """Observations are {image, pixel}; reject tiny baselines and inconsistent pixels."""
    seen = set()
    rows, usable = [], []
    for observation in observations:
        image = observation["image"]
        if image in seen or image not in cameras:
            continue
        seen.add(image)
        camera = cameras[image]
        pixel = vector(observation["pixel"], 2)
        if not 0 <= pixel[0] < camera.width or not 0 <= pixel[1] < camera.height:
            raise ValueError("Observation outside registered image")
        xy = cv2.undistortPoints(pixel.reshape(1, 1, 2), camera.K, camera.distortion).reshape(2)
        p = np.column_stack((camera.R, camera.t))
        rows.extend([xy[0] * p[2] - p[0], xy[1] * p[2] - p[1]])
        usable.append((camera, pixel))
    if len(usable) < min_views:
        raise ValueError(f"Need {min_views} distinct registered views")
    _, _, vt = np.linalg.svd(rows)
    homogeneous = vt[-1]
    if abs(homogeneous[3]) < 1e-10:
        raise ValueError("Triangulation at infinity")
    point = homogeneous[:3] / homogeneous[3]
    errors, rays, distances = [], [], []
    for camera, pixel in usable:
        projected, depths = camera.project([point])
        if depths[0] <= 0:
            raise ValueError("Triangulated point behind camera")
        errors.append(float(np.linalg.norm(projected[0] - pixel)))
        ray = point-camera.center
        distances.append(np.linalg.norm(ray))
        rays.append(ray / np.linalg.norm(ray))
    angle = max(math.degrees(math.acos(float(np.clip(np.dot(a, b), -1, 1))))
                for i, a in enumerate(rays) for b in rays[i+1:])
    if angle < min_angle_deg:
        raise ValueError("Insufficient triangulation baseline")
    if max(errors) > max_error_px:
        raise ValueError("Inconsistent multi-view observations; reprojection error too high")
    rms = float(np.sqrt(np.mean(np.square(errors))))
    focal = float(np.median([c.K[0, 0] for c, _ in usable]))
    uncertainty = float(max(rms, 0.5) * np.median(distances) / (focal * math.sin(math.radians(angle))))
    return point, dict(views=len(usable), reprojection_rms_px=rms, max_error_px=max(errors),
                       max_ray_angle_deg=angle, uncertainty_units=uncertainty,
                       uncertainty_kind="Geometric proxy; not validated anatomical accuracy")


class Surface:
    def __init__(self, vertices, triangles):
        self.vertices = np.asarray(vertices, float)
        self.triangles = np.asarray(triangles, int)
        if (self.vertices.ndim != 2 or self.vertices.shape[1] != 3
                or not np.isfinite(self.vertices).all() or len(self.vertices) < 3
                or self.triangles.ndim != 2 or self.triangles.shape[1] != 3
                or len(self.triangles) == 0 or self.triangles.min() < 0
                or self.triangles.max() >= len(self.vertices)):
            raise ValueError("Invalid mesh surface")
        self.adjacency = [[] for _ in self.vertices]
        lengths = []
        for triangle in self.triangles:
            a, b, c = self.vertices[triangle]
            if np.linalg.norm(np.cross(b-a, c-a)) < 1e-12:
                raise ValueError("Degenerate mesh triangle")
            for i, j in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
                length = float(np.linalg.norm(self.vertices[i]-self.vertices[j]))
                self.adjacency[i].append((j, length))
                self.adjacency[j].append((i, length))
                lengths.append(length)
        self.spacing = float(np.median(lengths))
        self.used_vertices = np.unique(self.triangles)

    def nearest_vertex(self, point, max_distance=None):
        point = vector(point)
        distances = np.linalg.norm(self.vertices[self.used_vertices] - point, axis=1)
        position = int(np.argmin(distances))
        if max_distance is not None and distances[position] > max_distance:
            raise ValueError("Point too far from observed surface")
        return int(self.used_vertices[position]), float(distances[position])

    def shortest_path(self, start, end):
        """Dijkstra along observed mesh edges, not an exact continuous geodesic."""
        a, a_snap = self.nearest_vertex(start, self.spacing * 2)
        b, b_snap = self.nearest_vertex(end, self.spacing * 2)
        distances = {a: 0.0}
        parents = {}
        queue = [(0.0, a)]
        while queue:
            cost, node = heapq.heappop(queue)
            if cost > distances[node]:
                continue
            if node == b:
                path = [b]
                while path[-1] != a:
                    path.append(parents[path[-1]])
                path.reverse()
                return cost + a_snap + b_snap, self.vertices[path], dict(
                    algorithm="Mesh-edge Dijkstra approximation", endpoint_snap_units=a_snap+b_snap,
                    discretization_error="Unknown; graph paths can overestimate continuous geodesics")
            for neighbor, length in self.adjacency[node]:
                new = cost + length
                if new < distances.get(neighbor, float("inf")):
                    distances[neighbor] = new
                    parents[neighbor] = node
                    heapq.heappush(queue, (new, neighbor))
        raise ValueError("Surface endpoints belong to disconnected mesh components")

    def line_intersections(self, origin, direction, ray=False):
        """Möller–Trumbore intersection against actual triangles, without filling holes."""
        origin, direction = vector(origin), vector(direction)
        direction = direction / np.linalg.norm(direction)
        triangles = self.vertices[self.triangles]
        edge1, edge2 = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
        h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
        determinant = np.einsum("ij,ij->i", edge1, h)
        valid = np.abs(determinant) > 1e-10
        inverse = np.divide(1, determinant, out=np.zeros_like(determinant), where=valid)
        s = origin-triangles[:, 0]
        u = inverse * np.einsum("ij,ij->i", s, h)
        q = np.cross(s, edge1)
        v = inverse * (q @ direction)
        t = inverse * np.einsum("ij,ij->i", edge2, q)
        valid &= (u >= -1e-8) & (v >= -1e-8) & (u+v <= 1+1e-8)
        if ray:
            valid &= t > 1e-8
        distances = sorted(t[valid].tolist())
        unique = []
        for value in distances:
            if not unique or abs(value-unique[-1]) > self.spacing * 1e-5:
                unique.append(value)
        return [origin+value*direction for value in unique]

    def associate_pixel(self, camera, pixel):
        origin, direction = camera.ray(pixel)
        hits = self.line_intersections(origin, direction, ray=True)
        if not hits:
            raise ValueError("Pixel ray does not intersect the observed mesh")
        return hits[0]


def plane_pair_intersections(surface, normal_a, offset_a, normal_b, offset_b):
    a, b = vector(normal_a), vector(normal_b)
    direction = np.cross(a, b)
    if np.linalg.norm(direction) < 1e-8:
        raise ValueError("Reference planes are parallel")
    origin = np.linalg.lstsq(np.vstack([a, b]), np.array([offset_a, offset_b]), rcond=None)[0]
    return surface.line_intersections(origin, direction)
