"""CPU-only mesh preview inside the OpenCV interface; no OpenGL/GPU requirement."""

import cv2
import numpy as np


class MeshPreview:
    def __init__(self):
        self.vertices = None
        self.triangles = None
        self.center = np.zeros(3)
        self.basis = np.eye(3)
        self.yaw = 0
        self.pitch = 0

    def set_mesh(self, mesh, camera=None):
        # Decimation is only for drawing; measurements retain the original mesh.
        if len(mesh.triangles) > 3500:
            mesh = mesh.simplify_quadric_decimation(3500)
        self.vertices = np.asarray(mesh.vertices).copy()
        self.triangles = np.asarray(mesh.triangles).copy()
        self.center = np.median(self.vertices, axis=0)
        self.basis = camera.R.copy() if camera else np.eye(3)

    def render(self, landmarks=None, width=640, height=480):
        image = np.full((height, width, 3), (27, 24, 23), np.uint8)
        if self.vertices is None:
            cv2.putText(image, "No reconstructed head surface", (30, height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 1, cv2.LINE_AA)
            return image
        yaw, pitch = np.radians(self.yaw), np.radians(self.pitch)
        c, s = np.cos(yaw), np.sin(yaw)
        cy, sy = np.cos(pitch), np.sin(pitch)
        rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ np.array([[1,0,0], [0,cy,-sy], [0,sy,cy]])
        points = (self.vertices-self.center) @ self.basis.T @ rotation.T
        extent = max(np.ptp(points[:, 0]), np.ptp(points[:, 1]), 1e-8)
        scale = min(width, height)*0.82/extent
        pixels = np.rint(points[:, :2]*scale + [width/2, height/2]).astype(int)
        triangles = self.triangles
        normals = np.cross(points[triangles[:, 1]]-points[triangles[:, 0]], points[triangles[:, 2]]-points[triangles[:, 0]])
        norm = np.linalg.norm(normals, axis=1)
        shade = 0.35+0.65*np.abs(np.divide(normals[:, 2], norm, out=np.zeros_like(norm), where=norm>0))
        depth = points[triangles, 2].mean(axis=1)
        for index in np.argsort(depth)[::-1]:
            color = tuple(int(channel*shade[index]) for channel in (205, 172, 120))
            cv2.fillConvexPoly(image, pixels[triangles[index]], color, cv2.LINE_AA)
        for name, item in (landmarks or {}).items():
            if item.get("point_3d") is None or item.get("status") == "UNRESOLVED":
                continue
            if name not in ("G", "O", "V", "P", "FL", "FR", "left_tragus", "right_tragus", "left_superior_ear_root", "right_superior_ear_root"):
                continue
            point = (np.asarray(item["point_3d"])-self.center) @ self.basis.T @ rotation.T
            x, y = np.rint(point[:2]*scale+[width/2, height/2]).astype(int)
            color = (70,220,110) if item["status"] == "DETECTED" else (60,190,255)
            cv2.circle(image, (x,y), 4, color, -1, cv2.LINE_AA)
            cv2.putText(image, name, (x+6,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        return image
