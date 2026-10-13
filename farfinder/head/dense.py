"""CPU multi-view stereo and observed-surface meshing, without hole filling."""

import math

import cv2
import numpy as np
import open3d as o3d

from farfinder.head.colmap import ReconstructionError, head_support


def cpu_stereo(session, cameras, masks, sparse_points, cancel, progress):
    center = np.median(sparse_points, axis=0)
    names = list(cameras)
    pairs = set()
    for name in names:
        camera = cameras[name]
        vector = camera.center-center
        candidates = []
        for other in names:
            if other == name:
                continue
            second = cameras[other]
            v = second.center-center
            angle = math.degrees(math.acos(float(np.clip(np.dot(vector, v)/np.linalg.norm(vector)/np.linalg.norm(v), -1, 1))))
            rotation_angle = math.degrees(math.acos(float(np.clip((np.trace(second.R @ camera.R.T)-1)/2, -1, 1))))
            if 7 <= angle <= 28 and rotation_angle < 35:
                candidates.append((angle, other))
        for _, other in sorted(candidates)[:2]:
            pairs.add(tuple(sorted((name, other))))
    points, colors = [], []
    accepted_pairs = 0
    for index, (name_a, name_b) in enumerate(sorted(pairs)):
        if cancel.is_set():
            raise ReconstructionError("Reconstruction cancelled")
        progress(f"CPU stereo pair {index+1}/{len(pairs)}")
        a, b = cameras[name_a], cameras[name_b]
        if (a.width, a.height) != (b.width, b.height):
            continue
        size = a.width, a.height
        rotation = b.R @ a.R.T
        translation = b.t-rotation @ a.t
        try:
            r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(a.K, a.distortion, b.K, b.distortion,
                                                       size, rotation, translation,
                                                       flags=cv2.CALIB_ZERO_DISPARITY, alpha=1)
            # SGBM assumes horizontal, positive left-to-right disparity.
            vertical = abs(p2[1, 3]) > abs(p2[0, 3])
            if p2[1 if vertical else 0, 3] > 0:
                a, b = b, a
                name_a, name_b = name_b, name_a
                rotation = b.R @ a.R.T
                translation = b.t-rotation @ a.t
                r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(a.K, a.distortion, b.K, b.distortion,
                                                           size, rotation, translation,
                                                           flags=cv2.CALIB_ZERO_DISPARITY, alpha=1)
            map_a = cv2.initUndistortRectifyMap(a.K, a.distortion, r1, p1, size, cv2.CV_32FC1)
            map_b = cv2.initUndistortRectifyMap(b.K, b.distortion, r2, p2, size, cv2.CV_32FC1)
            image_a = cv2.imread(str(session.path / "images" / name_a))
            image_b = cv2.imread(str(session.path / "images" / name_b))
            left = cv2.remap(image_a, *map_a, cv2.INTER_LINEAR)
            right = cv2.remap(image_b, *map_b, cv2.INTER_LINEAR)
            mask_a = cv2.remap(masks[name_a], *map_a, cv2.INTER_NEAREST)
            mask_b = cv2.remap(masks[name_b], *map_b, cv2.INTER_NEAREST)
            original_left = left
            if vertical:
                # Rotate rectified vertical epipolar lines to horizontal for SGBM.
                # Counter-clockwise rotation maps original y disparity to +x disparity.
                left, right = np.rot90(left).copy(), np.rot90(right).copy()
                mask_a, mask_b = np.rot90(mask_a).copy(), np.rot90(mask_b).copy()
            gray_a, gray_b = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            stereo_width = gray_a.shape[1]
            disparities = min(256, max(16, ((stereo_width//2-16)//16)*16))
            settings = dict(numDisparities=disparities, blockSize=5, P1=8*25, P2=32*25,
                            uniquenessRatio=12, speckleWindowSize=80, speckleRange=2,
                            disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
            forward = cv2.StereoSGBM_create(minDisparity=0, **settings).compute(gray_a, gray_b).astype(float)/16
            reverse = cv2.StereoSGBM_create(minDisparity=-disparities, **settings).compute(gray_b, gray_a).astype(float)/16
            y, x = np.indices(forward.shape)
            corresponding_x = np.rint(x-forward).astype(int)
            clipped_x = np.clip(corresponding_x, 0, stereo_width-1)
            lr_error = np.abs(forward+reverse[y, clipped_x])
            gray_float = gray_a.astype(np.float32)
            variance = cv2.boxFilter(gray_float**2, -1, (7, 7))-cv2.boxFilter(gray_float, -1, (7, 7))**2
            valid = ((forward > 1) & (forward < disparities-1) & (mask_a > 0)
                     & (mask_b[y, clipped_x] > 0) & (corresponding_x >= 0)
                     & (corresponding_x < stereo_width) & (lr_error < 1.2) & (variance > 40))
            if vertical:
                forward, valid = np.rot90(forward, -1).copy(), np.rot90(valid, -1).copy()
            xyz = cv2.reprojectImageTo3D(forward.astype(np.float32), q)
            camera_points = xyz[valid] @ r1
            world = (camera_points-a.t) @ a.R
            finite = np.isfinite(world).all(axis=1) & (camera_points[:, 2] > 0)
            world, rgb = world[finite], original_left[valid][finite][:, ::-1]/255
            if len(world) > 300:
                points.append(world)
                colors.append(rgb)
                accepted_pairs += 1
        except cv2.error:
            continue
    if not points:
        return None, dict(backend="CPU OpenCV SGBM", attempted_pairs=len(pairs), accepted_pairs=0,
                          reason="No reliable stereo disparity; sparse observed-surface fallback only")
    points, colors = np.vstack(points), np.vstack(colors)
    # Bound memory before testing all cameras. This only subsamples observed data.
    if len(points) > 250000:
        selected = np.random.default_rng(0).choice(len(points), 250000, replace=False)
        points, colors = points[selected], colors[selected]
    keep, _ = head_support(points, cameras, masks)
    # Dense matches must also remain close to independently reconstructed sparse
    # head geometry. A few camera cones alone can admit distant false matches.
    from scipy.spatial import cKDTree
    sparse_tree = cKDTree(sparse_points)
    spacing = float(np.median(sparse_tree.query(sparse_points,k=2)[0][:,1]))
    distances = sparse_tree.query(points,k=1)[0]
    keep &= distances < max(spacing*4, np.linalg.norm(np.ptp(sparse_points,axis=0))*0.025)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points[keep])
    cloud.colors = o3d.utility.Vector3dVector(colors[keep])
    return cloud, dict(backend="CPU OpenCV SGBM", attempted_pairs=len(pairs), accepted_pairs=accepted_pairs,
                       left_right_consistency_px=1.2, mask_supported_points=int(np.count_nonzero(keep)),
                       accuracy="Unvalidated on real heads; texture and occlusion dependent")


def observed_mesh(cloud):
    if len(cloud.points) < 150:
        raise ReconstructionError("Insufficient observed head points to generate a surface")
    extent = float(np.linalg.norm(np.ptp(np.asarray(cloud.points), axis=0)))
    if not np.isfinite(extent) or extent <= 0:
        raise ReconstructionError("Degenerate head point cloud")
    cloud = cloud.voxel_down_sample(extent * 0.0025)
    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    if len(cloud.points) < 150:
        raise ReconstructionError("Too few points remain after outlier removal")
    spacing = float(np.median(cloud.compute_nearest_neighbor_distance()))
    if spacing <= 0:
        raise ReconstructionError("Invalid point spacing")
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=spacing*5, max_nn=40))
    try:
        cloud.orient_normals_consistent_tangent_plane(20)
    except RuntimeError:
        raise ReconstructionError("Could not orient observed surface normals")
    normals = np.asarray(cloud.normals)
    offsets = np.asarray(cloud.points)-np.median(np.asarray(cloud.points), axis=0)
    if np.median(np.sum(normals*offsets, axis=1)) < 0:
        cloud.normals = o3d.utility.Vector3dVector(-normals)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        cloud, o3d.utility.DoubleVector([spacing*1.5, spacing*2, spacing*2.8]))
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if len(triangles) == 0:
        raise ReconstructionError("No supported triangles; head texture/coverage inadequate")
    edges = np.stack([vertices[triangles[:, 0]]-vertices[triangles[:, 1]],
                      vertices[triangles[:, 1]]-vertices[triangles[:, 2]],
                      vertices[triangles[:, 2]]-vertices[triangles[:, 0]]], axis=1)
    # Prune bridges over unobserved gaps; do not run Poisson/hole filling.
    mesh.remove_triangles_by_mask(np.max(np.linalg.norm(edges, axis=2), axis=1) > spacing*4)
    mesh.remove_degenerate_triangles().remove_duplicated_triangles().remove_unreferenced_vertices()
    if len(mesh.triangles) < 100:
        raise ReconstructionError("Observed mesh too fragmented for measurement")
    mesh.compute_vertex_normals()
    components, counts, _ = mesh.cluster_connected_triangles()
    removed_components = int(np.count_nonzero(np.asarray(counts)<20))
    if removed_components:
        labels=np.asarray(components)
        mesh.remove_triangles_by_mask(np.asarray(counts)[labels]<20)
        mesh.remove_unreferenced_vertices()
        components,counts,_=mesh.cluster_connected_triangles()
    if len(mesh.triangles)<100:
        raise ReconstructionError("Too little connected observed surface remains after fragment removal")
    return cloud, mesh, dict(method="Open3D ball pivoting over measured points; no hole filling",
                             points=len(cloud.points), vertices=len(mesh.vertices), triangles=len(mesh.triangles),
                             point_spacing_units=spacing, connected_components=len(counts),
                             largest_component_fraction=float(max(counts)/sum(counts)),
                             tiny_components_removed=removed_components,
                             watertight=bool(mesh.is_watertight()),
                             surface_status="PARTIAL_OBSERVED_SURFACE",
                             scalp_status="Hair/skin outer surface; underlying scalp is not observed through hair")
