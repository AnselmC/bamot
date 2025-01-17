import logging
from typing import Dict

import g2o
import numpy as np
from bamot.config import CONFIG as config
from bamot.core.base_types import ImageId, ObjectTrack, StereoCamera
from bamot.util.cv import from_homogeneous, to_homogeneous

LOGGER = logging.getLogger("CORE:OPTIMIZATION")

EmptyTuple = (None, None)


def get_obs_count(img_id, track):
    num_obs = 0
    for lm in track.landmarks.values():
        for obs in lm.observations:
            if obs.img_id == img_id:
                num_obs += 1
    return num_obs


def object_bundle_adjustment(
    object_track: ObjectTrack,
    all_poses: Dict[ImageId, np.ndarray],
    stereo_cam: StereoCamera,
    median_translation: float,
    max_iterations: int = 10,
    full_ba: bool = False,
) -> ObjectTrack:
    # setup optimizer
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
    algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(algorithm)
    added_poses: Dict[ImageId, int] = {}
    pose_id = 0  # even numbers
    prev_cam, prev_prev_cam = EmptyTuple, EmptyTuple
    const_motion_edges = []
    frames = list(object_track.poses.keys())
    max_speed = (
        config.MAX_SPEED_CAR if object_track.cls == "car" else config.MAX_SPEED_PED
    )
    max_dist = max_speed / config.FRAME_RATE
    if config.BA_NORMALIZE_TRANS_ERROR:
        trans_func = lambda x: 0.5 * (np.tanh(4 * x / max_dist - 2) + 1)
    else:
        trans_func = lambda x: 1

    for img_id in frames[-config.SLIDING_WINDOW_BA :] if not full_ba else frames:
        num_obs = get_obs_count(img_id, object_track)
        if num_obs < 5:
            # no associated landmarks, skip pose
            # reset prev and prev_prev_cam
            prev_cam, prev_prev_cam = EmptyTuple, EmptyTuple
            continue
        T_world_obj = object_track.poses[img_id]
        T_world_cam = all_poses[img_id]
        cam_params = stereo_cam.left
        T_obj_cam = np.linalg.inv(T_world_obj) @ T_world_cam
        pose_cam_params = g2o.ParameterSE3Offset()
        pose_cam_params.set_offset(g2o.Isometry3d(T_world_cam))
        pose_cam_params.set_id(img_id)
        optimizer.add_parameter(pose_cam_params)
        pose = g2o.Isometry3d(T_obj_cam)
        sba_cam = g2o.SBACam(pose.orientation(), pose.position())
        baseline = stereo_cam.T_left_right[0, 3]
        sba_cam.set_cam(
            cam_params.fx, cam_params.fy, cam_params.cx, cam_params.cy, baseline
        )
        pose_vertex = g2o.VertexCam()
        pose_vertex.set_id(pose_id)
        pose_vertex.set_estimate(sba_cam)
        pose_vertex.set_fixed(pose_id == 0)
        if config.USING_CONSTANT_MOTION:
            rot_weight, trans_weight = (
                config.CONSTANT_MOTION_WEIGHTS_CAR
                if object_track.cls == "car"
                else config.CONSTANT_MOTION_WEIGHTS_PED
            )
            rot_weight = rot_weight * num_obs
            trans_weight = trans_weight * num_obs * trans_func(median_translation)
            if prev_prev_cam == EmptyTuple:
                prev_prev_cam = (pose_vertex, img_id)
            elif prev_cam == EmptyTuple:
                prev_cam = (pose_vertex, img_id)
            else:
                const_motion_edge = g2o.EdgeSBALinearMotion()
                const_motion_edge.set_vertex(0, prev_prev_cam[0])
                const_motion_edge.set_vertex(1, prev_cam[0])
                const_motion_edge.set_vertex(2, pose_vertex)
                const_motion_edge.set_parameter_id(0, prev_prev_cam[1])
                const_motion_edge.set_parameter_id(1, prev_cam[1])
                const_motion_edge.set_parameter_id(2, img_id)
                info = np.diag(
                    np.hstack([np.repeat(trans_weight, 3), np.repeat(rot_weight, 3),])
                )
                const_motion_edge.set_information(info)
                const_motion_edges.append(const_motion_edge)
                # robust_kernel = g2o.RobustKernelHuber()
                # const_motion_edge.set_robust_kernel(robust_kernel)
                optimizer.add_edge(const_motion_edge)
                prev_prev_cam = prev_cam
                prev_cam = (pose_vertex, img_id)
        added_poses[img_id] = pose_id
        pose_id += 2
        optimizer.add_vertex(pose_vertex)
    landmark_id = 1  # odd numbers
    landmark_mapping = {}
    mono_edges = []
    stereo_edges = []
    # iterate over all landmarks of object
    num_landmarks = 0
    # optimize over all landmarks
    for idx, landmark in object_track.landmarks.items():
        # add landmark as vertex
        point_vertex = g2o.VertexSBAPointXYZ()
        point_vertex.set_id(landmark_id)
        landmark_mapping[idx] = landmark_id
        point_vertex.set_marginalized(True)
        point_vertex.set_estimate(landmark.pt_3d.reshape(3,))
        landmark_id += 2
        optimizer.add_vertex(point_vertex)
        num_landmarks += 1
        # optimize over all observations
        for obs in landmark.observations:
            if obs.img_id not in added_poses.keys():
                continue
            # add edge between landmark and cam
            # feature coordinates of observation are x_i_t
            measurement = obs.pt_2d
            if measurement.shape == (3,):
                edge = g2o.EdgeProjectP2SC()
                info = np.identity(3)
                delta = 1.5
                stereo_edges.append((edge, idx))

            else:
                edge = g2o.EdgeProjectP2MC()
                info = np.identity(2)
                delta = 1
                mono_edges.append((edge, idx))

            edge.set_vertex(0, point_vertex)
            edge.set_vertex(1, optimizer.vertex(added_poses[obs.img_id]))
            edge.set_measurement(measurement)
            edge.set_information(info)
            robust_kernel = g2o.RobustKernelHuber()
            robust_kernel.set_delta(delta)
            edge.set_robust_kernel(robust_kernel)
            optimizer.add_edge(edge)
    if not mono_edges and not stereo_edges and not const_motion_edges:
        LOGGER.debug("Nothing to optimize...")
        return object_track
    # optimize
    LOGGER.debug(
        "Starting optimization w/ %d landmarks, %d poses, and %d observations",
        num_landmarks,
        len(added_poses),
        len(mono_edges) + len(stereo_edges),
    )
    optimizer.initialize_optimization(0)
    optimizer.set_verbose(logging.getLogger().level == logging.DEBUG)
    optimizer.optimize(max_iterations)
    num_outliers = 0
    for edge, _ in mono_edges:
        edge.compute_error()
        pt_obj, T_obj_cam = edge.vertices()
        pt_obj = pt_obj.estimate().reshape(3, 1)
        T_obj_cam = T_obj_cam.estimate().matrix()
        pczi = (
            1.0
            / from_homogeneous(np.linalg.inv(T_obj_cam) @ to_homogeneous(pt_obj))[-1]
            ** 2
        )
        if edge.chi2() > 1 or not np.isfinite(pczi):
            edge.set_level(1)
            num_outliers += 1
        edge.set_robust_kernel(None)
    for edge, _ in stereo_edges:
        edge.compute_error()
        pt_obj, T_obj_cam = edge.vertices()
        pt_obj = pt_obj.estimate().reshape(3, 1)
        T_obj_cam = T_obj_cam.estimate().matrix()
        pt_cam = from_homogeneous(np.linalg.inv(T_obj_cam) @ to_homogeneous(pt_obj))
        if edge.chi2() > 2 or np.isinf(1.0 / pt_cam[2] ** 2):
            edge.set_level(1)
            num_outliers += 1
        edge.set_robust_kernel(None)

    # optimize without outliers
    LOGGER.debug("Starting optimization w/o %d outliers", num_outliers)
    optimizer.initialize_optimization(0)
    optimizer.optimize(max_iterations)
    num_outliers = 0
    landmarks_to_remove = set()
    for edge, lmid in mono_edges:
        edge.compute_error()
        pt_obj, T_obj_cam = edge.vertices()
        pt_obj = pt_obj.estimate().reshape(3, 1)
        T_obj_cam = T_obj_cam.estimate().matrix()
        pt_cam = from_homogeneous(np.linalg.inv(T_obj_cam) @ to_homogeneous(pt_obj))
        if edge.chi2() > 1 or np.isinf(1.0 / pt_cam[2] ** 2):
            num_outliers += 1
            landmarks_to_remove.add(lmid)
    for edge, lmid in stereo_edges:
        edge.compute_error()
        pt_obj, T_obj_cam = edge.vertices()
        pt_obj = pt_obj.estimate().reshape(3, 1)
        T_obj_cam = T_obj_cam.estimate().matrix()
        pt_cam = from_homogeneous(np.linalg.inv(T_obj_cam) @ to_homogeneous(pt_obj))
        if edge.chi2() > 2 or np.isinf(1.0 / pt_cam[2] ** 2):
            num_outliers += 1
            landmarks_to_remove.add(lmid)
    # update landmark positions of objects (w.r.t. to object) and object poses over time
    for landmark_idx, vertex_idx in landmark_mapping.items():
        updated_point = optimizer.vertex(vertex_idx).estimate().reshape(3, 1)
        object_track.landmarks[landmark_idx].pt_3d = updated_point.copy()
    LOGGER.debug("Removing %d landmarks", len(landmarks_to_remove))
    for lmid in landmarks_to_remove:
        object_track.landmarks.pop(lmid)
    for img_id, vertex_idx in added_poses.items():
        T_obj_cam = optimizer.vertex(vertex_idx).estimate().matrix()
        T_world_cam = all_poses[img_id]
        T_world_obj = T_world_cam @ np.linalg.inv(T_obj_cam)
        object_track.poses[img_id] = T_world_obj.copy()
    return object_track
