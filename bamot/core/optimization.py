import logging
from typing import Dict

import g2o
import numpy as np

from bamot.core.base_types import ImageId, ObjectTrack, StereoCamera, TimeCamId
from bamot.util.cv import from_homogeneous_pt, to_homogeneous_pt

LOGGER = logging.getLogger("CORE:OPTIMIZATION")


def object_bundle_adjustment(
    object_track: ObjectTrack,
    all_poses: Dict[ImageId, np.ndarray],
    stereo_cam: StereoCamera,
    max_iterations: int = 10,
) -> ObjectTrack:
    # setup optimizer
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
    algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(algorithm)
    added_poses: Dict[TimeCamId, int] = {}
    pose_id = 0  # even numbes
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
        # print(landmark.pt_3d)
        landmark_id += 2
        optimizer.add_vertex(point_vertex)
        num_landmarks += 1
        # optimize over all observations
        for obs in landmark.observations:
            # x_t_i is the 2d pt of an observation (i.e. landmark i at frame t) -> i.e. feature location
            # X_q_i is the landmark in object coordinates (doesn't vary over t)
            # T_c_q is the object pose at frame t (T_cam_obj)
            img_id = obs.timecam_id[0]
            left = obs.timecam_id[1] == 0
            T_world_left = all_poses[img_id]
            if left:
                T_world_cam = T_world_left
                params = stereo_cam.left
            else:
                T_world_cam = T_world_left @ stereo_cam.T_left_right
                params = stereo_cam.right
            T_world_obj = object_track.poses[obs.timecam_id[0]]
            # get pose for observations, i.e. camera pose * object_pose
            T_obj_cam = np.linalg.inv(T_world_obj) @ T_world_cam
            # print(T_obj_cam)
            # print(landmark.pt_3d)
            # pt_cam = from_homogeneous_pt(
            #    np.linalg.inv(T_obj_cam) @ to_homogeneous_pt(landmark.pt_3d)
            # )
            # print(pt_cam)
            # print(1 / pt_cam[0] ** 2)
            # add pose to graph if it hasn't been added yet
            if obs.timecam_id not in added_poses.keys():
                pose = g2o.Isometry3d(T_obj_cam)
                sba_cam = g2o.SBACam(pose.orientation(), pose.position())
                baseline = 0.5
                sba_cam.set_cam(params.fx, params.fy, params.cx, params.cy, baseline)
                pose_vertex = g2o.VertexCam()
                pose_vertex.set_id(pose_id)
                pose_vertex.set_estimate(sba_cam)
                pose_vertex.set_fixed(pose_id == 0)  # fix first cam
                added_poses[obs.timecam_id] = pose_id
                pose_id += 2
                optimizer.add_vertex(pose_vertex)

            # add edge between landmark and cam
            # feature coordinates of observation are x_i_t
            measurement = obs.pt_2d
            if measurement.shape == (3,):
                edge = g2o.EdgeProjectP2SC()
                info = np.identity(3)
                delta = 7.815
                stereo_edges.append((edge, idx))

            else:
                edge = g2o.EdgeProjectP2MC()
                info = np.identity(2)
                delta = 5.991
                mono_edges.append((edge, idx))

            edge.set_vertex(0, point_vertex)
            edge.set_vertex(1, optimizer.vertex(added_poses[obs.timecam_id]))
            edge.set_measurement(obs.pt_2d)
            edge.set_information(info)
            robust_kernel = g2o.RobustKernelHuber()
            robust_kernel.set_delta(delta)
            edge.set_robust_kernel(robust_kernel)
            optimizer.add_edge(edge)
    # optimize
    LOGGER.debug(
        "Starting optimization w/ %d landmarks, %d poses, and %d observations",
        num_landmarks,
        len(added_poses),
        len(mono_edges) + len(stereo_edges),
    )
    optimizer.initialize_optimization(0)
    optimizer.set_verbose(True)
    optimizer.optimize(max_iterations)
    num_outliers = 0
    for edge, _ in mono_edges:
        edge.compute_error()
        pt_obj, T_obj_cam = edge.vertices()
        pt_obj = pt_obj.estimate()
        T_obj_cam = T_obj_cam.estimate().matrix()
        pczi = (
            1.0
            / from_homogeneous_pt(np.linalg.inv(T_obj_cam) @ to_homogeneous_pt(pt_obj))[
                -1
            ]
            ** 2
        )
        if edge.chi2() > 5.991 or np.isinf(pczi) or np.isnan(pczi):
            edge.set_level(1)
            num_outliers += 1
        edge.set_robust_kernel(None)
    for edge, _ in stereo_edges:
        edge.compute_error()
        pt_obj, T_obj_cam = edge.vertices()
        pt_obj = pt_obj.estimate()
        T_obj_cam = T_obj_cam.estimate().matrix()
        pt_cam = from_homogeneous_pt(
            np.linalg.inv(T_obj_cam) @ to_homogeneous_pt(pt_obj)
        )
        if edge.chi2() > 7.815 or np.isinf(1.0 / pt_cam[2] ** 2):
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
        pt_obj = pt_obj.estimate()
        T_obj_cam = T_obj_cam.estimate().matrix()
        pt_cam = from_homogeneous_pt(
            np.linalg.inv(T_obj_cam) @ to_homogeneous_pt(pt_obj)
        )
        if edge.chi2() > 5.991 or np.isinf(1.0 / pt_cam[2] ** 2):
            num_outliers += 1
            landmarks_to_remove.add(lmid)
    for edge, lmid in stereo_edges:
        edge.compute_error()
        pt_obj, T_obj_cam = edge.vertices()
        pt_obj = pt_obj.estimate()
        T_obj_cam = T_obj_cam.estimate().matrix()
        pt_cam = from_homogeneous_pt(
            np.linalg.inv(T_obj_cam) @ to_homogeneous_pt(pt_obj)
        )
        if edge.chi2() > 7.815 or np.isinf(1.0 / pt_cam[2] ** 2):
            num_outliers += 1
            landmarks_to_remove.add(lmid)
    # update landmark positions of objects (w.r.t. to object) and object poses over time
    for landmark_idx, vertex_idx in landmark_mapping.items():
        # print("Updating point from ")
        # print(object_track.landmarks[landmark_idx].pt_3d)
        updated_point = optimizer.vertex(vertex_idx).estimate().reshape(3, 1)
        object_track.landmarks[landmark_idx].pt_3d = updated_point.copy()
        # print("to ")
        # print(object_track.landmarks[landmark_idx].pt_3d)
    LOGGER.debug("Removing %d landmarks", len(landmarks_to_remove))
    for lmid in landmarks_to_remove:
        object_track.landmarks.pop(lmid)
    for timecam_id, vertex_idx in added_poses.items():
        T_obj_cam = optimizer.vertex(vertex_idx).estimate().matrix().copy()
        img_id = timecam_id[0]
        left = timecam_id[1] == 0
        T_world_left = all_poses[img_id]
        if left:
            T_world_cam = T_world_left
        else:
            T_world_cam = T_world_left @ stereo_cam.T_left_right
        T_world_obj = T_world_cam @ np.linalg.inv(T_obj_cam)
        print("Updating pose from")
        print(object_track.poses[timecam_id[0]])
        object_track.poses[timecam_id[0]] = T_world_obj
        print("to")
        print(object_track.poses[timecam_id[0]])
    return object_track
