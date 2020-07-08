from typing import Dict, Tuple

import numpy as np

import g2o
from bamot.core.base_types import (Camera, CameraParameters, ObjectTrack,
                                   TimeCamId)
from bamot.util.cv import to_homogeneous_pt

# everything is happening in camera coordinates


def object_bundle_adjustment(
    object_track: ObjectTrack,
    cameras: Dict[TimeCamId, Tuple[np.ndarray, CameraParameters]],
    max_iterations: int = 10,
):
    # setup optimizer
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
    algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(algorithm)
    # optimize over all landmarks
    added_poses: Dict[TimeCamId, int] = {}
    pose_id = 0  # even numbers
    landmark_id = 1  # odd numbers
    landmark_mapping = {}
    # iterate over all landmarks of object
    for idx, landmark in enumerate(object_track.landmarks):
        # add landmark as vertex
        # optimize over all observations
        vertex_point = g2o.VertexSBAPointXYZ()
        vertex_point.set_id(landmark_id)
        landmark_mapping[idx] = landmark_id
        vertex_point.set_marginalized(True)
        # pt_3d_obj = landmark.pt_3d
        # pt_3d_world = T_world_obj @ to_homogeneous_pt(pt_3d_obj)
        vertex_point.set_estimate(landmark.pt_3d)
        landmark_id += 1
        optimizer.add_vertex()
        # go over all observations of landmark
        for obs in landmark.observations:
            # x_t_i is the 2d pt of an observation (i.e. landmark i at frame t) -> i.e. feature location
            # X_q_i is the landmark in object coordinates (doesn't vary over t)
            # T_c_q is the object pose at frame t (T_cam_obj)
            T_world_cam, params = cameras[obs.timecam_id]
            T_world_obj = object_track.poses[obs.timecam_id[0]]
            # get pose for observations, i.e. camera pose * object_pose
            T_cam_obj = np.linalg.inv(T_world_cam) @ T_world_obj
            # add pose to graph if it hasn't been added yet
            if obs.timecam_id not in added_poses.keys():
                pose = g2o.Isometry3d(T_cam_obj)
                sba_cam = g2o.SBACam(pose.orientation(), pose.position())
                sba_cam.set_cam(params.fx, params.fy, params.cx, params.cy, 0.0)
                pose_vertex = g2o.VertexCam()
                pose_vertex.set_id(pose_id)
                pose_vertex.set_estimate(sba_cam)
                pose_vertex.set_fixed(pose_id == 0)  # fix first cam
                added_poses[obs.timecam_id] = pose_id
                pose_id += 2
                optimizer.add_vertex(pose_vertex)

            # add edge between landmark and cam
            # feature coordinates of observation are x_i_t
            edge = g2o.EdgeProjectXYZ()
            edge.set_vertex(0, vertex_point)
            edge.set_vertex(1, optimizer.vertex(added_poses[obs.timecam_id]))
            edge.set_measurement(obs.pt_2d)
            edge.set_information(np.identity(2))
            edge.set_robust_kernel(g2o.RobustKernelHuber())
            optimizer.add_edge(edge)
    # optimize
    optimizer.initialize_optimization()
    optimizer.set_verbose(True)
    optimizer.optimizer(max_iterations)
    # update landmark positions of objects (w.r.t. to object) and object poses over time
    for landmark_idx, vertex_idx in landmark_mapping.items():
        object_track.landmarks[landmark_idx].pt_3d = (
            optimizer.vertex(vertex_idx).estimate().matrix()
        )
    for timecam_id, vertex_idx in added_poses.items():
        T_cam_obj = optimizer.vertex(vertex_idx).estimate().matrix()
        T_world_cam, _ = cameras[obs.timecam_id]
        # T_world_obj = T_world_cam @ T_cam_obj
        object_track.poses[timecam_id[0]] = T_world_cam @ T_cam_obj
    return object_track
