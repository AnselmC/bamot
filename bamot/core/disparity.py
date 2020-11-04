import copy
import logging
import time

import numpy as np

import cv2
from bamot.core.base_types import (Landmark, ObjectTrack, StereoImage,
                                   TrackMatch)
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
    Tx = -stereo_cam.T_left_right[0, 3]
    f = stereo_cam.left.fx  # should this be sqrt(2) * f?

    return np.array(
        [
            [1, 0, 0, -cx],
            [0, 1, 0, -cy],
            [0, 0, 0, f],
            [0, 0, -1 / Tx, (cx - cx2) / Tx],
        ]
    )


def get_entire_point_cloud(disp, Q):
    # need to divide by 16 and cast to float as per:
    # https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga1bc1152bd57d63bc524204f21fde6e02
    # also, returned points are in left camera coordinates
    points_3d = cv2.reprojectImageTo3D((disp / 16).astype(np.float32), Q)
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


def create_disparity_stereo_img(stereo_img, disp):
    gray_left, _ = map(
        lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY),
        [stereo_img.left, stereo_img.right],
    )
    disp = normalize_img(disp)
    return StereoImage(left=gray_left, right=disp)


# def create_landmarks_from_pointcloud(point_cloud, T_obj_cam):
def create_landmarks_from_pointcloud(point_cloud_obj):
    landmarks = {}
    # T_obj_world is always T_world_cam + cluster_center_mean
    for i, pt_3d in enumerate(point_cloud_obj):
        # x, y, z = from_homogeneous_pt(T_obj_cam @ to_homogeneous_pt(pt_3d))
        # x, y, z = -pt_3d
        # landmarks[i] = Landmark(np.array([-y, z, -x]).reshape(3, 1), [])
        landmarks[i] = Landmark(pt_3d.reshape(3, 1), [])
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
    disp_computer = cv2.StereoBM_create(numDisparities=48, blockSize=23)
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
        active_tracks = []
        for match in matches:
            active_tracks.append(match.track_index)
            if object_tracks.get(match.track_index) is None:
                LOGGER.debug("Added track with index %d", match.track_index)
                object_tracks[match.track_index] = ObjectTrack(
                    landmarks={},
                    poses={img_id: current_pose},
                    cls=new_detections[match.detection_index].left.cls,
                )
            track = object_tracks[match.track_index]
            detection = new_detections[match.detection_index]
            # mask out point cloud
            full_mask = valid_mask & detection.left.mask & valid_dist
            # point cloud is in left camera coordinates
            point_cld_cam = point_cld[full_mask]
            # move object coordinate system to center/mean of cluster
            T_world_cam = current_pose
            T_world_obj = current_pose.copy()
            t_world_cam = T_world_cam[:3, 3].reshape(3, 1).copy()
            cluster_center = np.mean(point_cld_cam, axis=0)
            t_cam_obj = -cluster_center.reshape(3, 1)
            T_cam_obj = np.identity(4)
            T_cam_obj[:3, 3] = t_cam_obj.reshape(3)
            T_world_obj[:3, 3] = (t_world_cam + t_cam_obj).reshape(3)
            # transform point_cld_cam to obj coordinates
            # create homogeneous point_cloud
            # point_cld_cam_hom = np.ones((point_cld_cam.shape[0], 4))
            # point_cld_cam_hom[:, :3] = point_cld_cam.copy()
            # T_obj_cam = np.linalg.inv(T_world_obj) @ T_world_cam
            # point_cld_obj_hom = T_obj_cam @ point_cld_cam_hom.T
            # point_cld_obj = (point_cld_obj_hom[:, :3].T / point_cld_obj_hom[:3, 3]).T
            # point_cld_obj = point_cld_cam - cluster_center
            # point_cld_obj = point_cld_cam - cluster_center
            # create transformation from obj to world
            # T_world_cam = current_pose
            # t_world_obj = T_world_cam[:3, 3] - cluster_center
            # T_world_obj = np.identity(4)
            # T_world_obj[:3, 3] = t_world_obj.reshape(3)
            # add pose to track
            track.poses[img_id] = T_world_obj
            track.poses[img_id] = T_world_cam
            track.landmarks = create_landmarks_from_pointcloud(point_cld_cam)

        old_tracks = set(object_tracks.keys()).difference(set(active_tracks))
        for track_id in old_tracks:
            if object_tracks[track_id].active:
                object_tracks[track_id].active = False
        shared_data.put(
            {
                "object_tracks": copy.deepcopy(object_tracks),
                "stereo_image": create_disparity_stereo_img(stereo_img, disp),
                "img_id": img_id,
                "current_cam_pose": current_pose,
            }
        )
    stop_flag.set()
    shared_data.put({})  # final data to eliminate race condition
    returned_data.put(_compute_estimated_trajectories(object_tracks, all_poses))
