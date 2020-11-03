import copy
import logging
import time

import numpy as np

import cv2
from bamot.core.base_types import Landmark, ObjectTrack, TrackMatch
from bamot.core.mot import _compute_estimated_trajectories
from bamot.util.cv import from_homogeneous_pt, to_homogeneous_pt

LOGGER = logging.getLogger("CORE:DISPARITY")


def compute_disparity(stereo_img, disparity_computer):
    gray_left, gray_right = map(
        lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY),
        [stereo_img.left, stereo_img.right],
    )
    disparity = disparity_computer.compute(gray_left, gray_right)
    return disparity


def get_Q_matrix(stereo_cam):
    cx = stereo_cam.left.cx
    cx2 = stereo_cam.right.cx
    cy = stereo_cam.left.cy
    Tx = 10 * stereo_cam.T_left_right[0, 3]  # TODO: why factor of 10?
    f = stereo_cam.left.fx
    return np.array(
        [
            [1, 0, 0, -cx],
            [0, 1, 0, -cy],
            [0, 0, 0, f],
            [0, 0, -1 / Tx, (cx - cx2) / Tx],
        ]
    )


def get_entire_point_cloud(disp, Q):
    points_3d = cv2.reprojectImageTo3D(disp, Q)
    return points_3d


def normalize_img(img):
    return (255 * (img - img.min()) / (img.max() - img.min())).astype(np.uint8)


def display_result(stereo_img, disp):
    gray_left, _ = map(
        lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY),
        [stereo_img.left, stereo_img.right],
    )
    disp = normalize_img(disp)
    cv2.imshow("res", np.vstack([gray_left, disp]))
    cv2.waitKey(0)


def display_disparity(disp):
    disp = normalize_img(disp)
    cv2.imshow("disp", disp)
    cv2.waitKey(0)


def create_landmarks_from_pointcloud(point_cloud, T_obj_cam):
    landmarks = {}
    # T_obj_world is always T_world_cam + cluster_center_mean
    for i, pt_3d in enumerate(point_cloud):
        x, y, z = from_homogeneous_pt(T_obj_cam @ to_homogeneous_pt(pt_3d))
        landmarks[i] = Landmark(np.array([x, y, z]).reshape(3, 1), [])
    return landmarks


def run(
    images,
    detections,
    stereo_cam,
    all_poses,
    shared_data,
    stop_flag,
    returned_data,
    next_step,
    continuous,
):
    disp_computer = cv2.StereoBM_create(numDisparities=48)
    Q = get_Q_matrix(stereo_cam)
    object_tracks = {}
    for (img_id, stereo_img), new_detections in zip(images, detections):
        if stop_flag.is_set():
            break
        while not continuous and not next_step.is_set():
            time.sleep(0.05)
        next_step.clear()
        current_pose = all_poses[img_id]
        disp = compute_disparity(stereo_img, disp_computer)
        point_cld = get_entire_point_cloud(disp, Q)
        valid_mask = disp > disp.min()
        valid_dist = np.isfinite(np.linalg.norm(point_cld, axis=2))
        matches = [
            TrackMatch(track_index=track_idx, detection_index=detection_idx)
            for detection_idx, track_idx in enumerate(
                map(lambda x: x.left.track_id, new_detections)
            )
        ]
        for match in matches:
            if object_tracks.get(match.track_index) is None:
                LOGGER.debug("Added track with index %d", match.track_index)
                object_tracks[match.track_index] = ObjectTrack(
                    landmarks={},
                    poses={img_id: current_pose},
                    cls=new_detections[match.detection_index].left.cls,
                )
            detection = new_detections[match.detection_index]
            full_mask = valid_mask & detection.left.mask & valid_dist
            point_cld_obj = point_cld[full_mask]
            track = object_tracks[match.track_index]
            cluster_center = np.mean(point_cld_obj, axis=0)
            T_world_cam = current_pose
            t_cam_obj = cluster_center
            t_world_obj = from_homogeneous_pt(
                T_world_cam @ to_homogeneous_pt(t_cam_obj)
            )
            T_world_obj = np.identity(4)
            T_world_obj[:3, 3] = t_world_obj.reshape(3)
            # T_world_obj = np.linalg.inv(current_pose)
            track.poses[img_id] = T_world_obj
            T_obj_cam = np.linalg.inv(T_world_obj) @ T_world_cam
            track.landmarks = create_landmarks_from_pointcloud(point_cld_obj, T_obj_cam)

        shared_data.put(
            {
                "object_tracks": copy.deepcopy(object_tracks),
                "stereo_image": stereo_img,
                "img_id": img_id,
                "current_cam_pose": current_pose,
            }
        )
    stop_flag.set()
    shared_data.put({})  # final data to eliminate race condition
    returned_data.put(_compute_estimated_trajectories(object_tracks, all_poses))
