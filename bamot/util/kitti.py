import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from bamot.core.base_types import (CameraParameters, ObjectDetection,
                                   StereoCamera)
from bamot.util.cv import from_homogeneous_pt, to_homogeneous_pt

LOGGER = logging.getLogger("Util:Kitti")


def _project(pt_3d_cam: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    pt_3d_cam = pt_3d_cam[:4] / pt_3d_cam[3]
    pt_2d_hom = intrinsics[:3, :3] @ pt_3d_cam[:3]
    return pt_2d_hom


def _back_project(pt_2d: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    pt_2d = pt_2d.reshape(2, 1)
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    u, v = map(float, pt_2d)
    mx = (u - cx) / fx
    my = (v - cy) / fy
    length = np.sqrt(mx ** 2 + my ** 2 + 1)
    return (np.array([mx, my, 1]) / length).reshape(3, 1)


TrackIdToPoseDict = Dict[int, Dict[int, np.ndarray]]
TrackIdToOcclusionLevel = Dict[int, Dict[int, int]]


def get_trajectories_from_kitti(
    detection_file: Path, poses: List[np.ndarray], offset: int, T02: np.ndarray
) -> Tuple[TrackIdToPoseDict, TrackIdToPoseDict, TrackIdToOcclusionLevel]:
    gt_trajectories_world = {}
    gt_trajectories_cam = {}
    occlusion_levels = {}
    if not detection_file.exists():
        return gt_trajectories_world, gt_trajectories_cam
    with open(detection_file.as_posix(), "r") as fp:
        for line in fp:
            cols = line.split(" ")
            frame = int(cols[0])
            if frame < offset:
                continue
            track_id = int(cols[1])
            if track_id == -1:
                continue
            # cols[2] is type (car, ped, ...)
            # cols[3] is level of truncation
            # cols[4] is level of occlusion
            occlusion_level = int(cols[4])
            # cols[5] is observation angle of object
            # cols[6:10] is bbox
            # cols[10: 13] are 3D dimensions
            location_cam0 = np.array(list(map(float, cols[13:16]))).reshape(
                (3, 1)
            )  # in camera coordinates
            T_w_cam2 = poses[frame]
            location_cam2 = from_homogeneous_pt(
                np.linalg.inv(T02) @ to_homogeneous_pt(location_cam0)
            )
            location_world = from_homogeneous_pt(
                T_w_cam2 @ to_homogeneous_pt(location_cam2)
            )
            # cols[16] is rotation of object
            # cols[17] is score
            if gt_trajectories_world.get(track_id) is None:
                gt_trajectories_world[track_id] = {}
                gt_trajectories_cam[track_id] = {}
                occlusion_levels[track_id] = {}
            gt_trajectories_world[track_id][frame] = location_world.tolist()
            gt_trajectories_cam[track_id][frame] = location_cam2.tolist()
            occlusion_levels[track_id][frame] = occlusion_level
    LOGGER.debug("Extracted GT trajectories for %d objects", len(gt_trajectories_world))
    return gt_trajectories_world, gt_trajectories_cam, occlusion_levels


def get_gt_poses_from_kitti(oxts_file: Path) -> List[np.ndarray]:
    """Adapted from Sergio Agostinho, returns GT poses of left cam
    """
    poses = []
    if not oxts_file.exists():
        return poses

    cols = (
        "lat",
        "lon",
        "alt",
        "roll",
        "pitch",
        "yaw",
        "vn",
        "ve",
        "vf",
        "vl",
        "vu",
        "ax",
        "ay",
        "az",
        "af",
        "al",
        "au",
        "wx",
        "wy",
        "wz",
        "wf",
        "wl",
        "wu",
        "posacc",
        "velacc",
        "navstat",
        "numsats",
        "posmode",
        "velmode",
        "orimode",
    )
    df = pd.read_csv(oxts_file.as_posix(), sep=" ", names=cols, index_col=False)

    # extract relative poses w.r.t. original frame (i.e. world)
    # get poses into expected frame
    # rotate around 90 degrees around x
    # rotation 90 degrees around z
    # rot_x = np.array(
    #    [
    #        [1, 0, 0],
    #        [0, np.cos(np.pi / 2), -np.sin(np.pi / 2)],
    #        [0, np.sin(np.pi / 2), np.cos(np.pi / 2)],
    #    ]
    # )
    # rot_z = np.array(
    #    [
    #        [np.cos(np.pi / 2), -np.sin(np.pi / 2), 0],
    #        [np.sin(np.pi / 2), np.cos(np.pi / 2), 0],
    #        [0, 0, 1],
    #    ]
    # )
    # rot_y = np.array([[np.cos(0), 0, np.sin(0)], [0, 1, 0], [-np.sin(0), 0, np.cos(0)]])
    # roll = np.identity(4)
    # roll[:3, :3] = rot_x

    # pitch = np.identity(4)
    # pitch[:3, :3] = rot_y

    # yaw = np.identity(4)
    # yaw[:3, :3] = rot_z
    rot = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    poses = [
        rot @ T_w_cam2
        for T_w_cam2 in _oxts_to_poses(
            *df[["lat", "lon", "alt", "roll", "pitch", "yaw"]].values.T
        )
    ]

    return poses


def _oxts_to_poses(lat, lon, alt, roll, pitch, yaw):
    """This implementation is a python reimplementation of the convertOxtsToPose
    MATLAB function in the original development toolkit for raw data
    """

    def rot_x(theta):
        theta = np.atleast_1d(theta)
        n = len(theta)
        return np.stack(
            (
                np.stack([np.ones(n), np.zeros(n), np.zeros(n)], axis=-1),
                np.stack([np.zeros(n), np.cos(theta), -np.sin(theta)], axis=-1),
                np.stack([np.zeros(n), np.sin(theta), np.cos(theta)], axis=-1),
            ),
            axis=-2,
        )

    def rot_y(theta):
        theta = np.atleast_1d(theta)
        n = len(theta)
        return np.stack(
            (
                np.stack([np.cos(theta), np.zeros(n), np.sin(theta)], axis=-1),
                np.stack([np.zeros(n), np.ones(n), np.zeros(n)], axis=-1),
                np.stack([-np.sin(theta), np.zeros(n), np.cos(theta)], axis=-1),
            ),
            axis=-2,
        )

    def rot_z(theta):
        theta = np.atleast_1d(theta)
        n = len(theta)
        return np.stack(
            (
                np.stack([np.cos(theta), -np.sin(theta), np.zeros(n)], axis=-1),
                np.stack([np.sin(theta), np.cos(theta), np.zeros(n)], axis=-1),
                np.stack([np.zeros(n), np.zeros(n), np.ones(n)], axis=-1),
            ),
            axis=-2,
        )

    n = len(lat)

    # converts lat/lon coordinates to mercator coordinates using mercator scale
    #        mercator scale             * earth radius
    scale = np.cos(lat[0] * np.pi / 180.0) * 6378137

    position = np.stack(
        [
            scale * lon * np.pi / 180.0,
            scale * np.log(np.tan((90.0 + lat) * np.pi / 360.0)),
            alt,
        ],
        axis=-1,
    )

    R = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)

    # extract relative transformation with respect to the first frame
    T0_inv = np.block([[R[0].T, -R[0].T @ position[0].reshape(3, 1)], [0, 0, 0, 1]])
    T = T0_inv @ np.block(
        [[R, position[:, :, None]], [np.zeros((n, 1, 3)), np.ones((n, 1, 1))]]
    )
    return T


def get_cameras_from_kitti(calib_file: Path) -> Tuple[StereoCamera, np.ndarray]:
    with open(calib_file.as_posix(), "r") as fp:
        for line in fp:
            cols = line.split(" ")
            name = cols[0]
            # if "R_02" in name:
            #    R02 = np.array(list(map(float, cols[1:]))).reshape(3, 3)
            # elif "T_02" in name:
            #    t02 = np.array(list(map(float, cols[1:]))).reshape(3)
            # elif "R_03" in name:
            #    R03 = np.array(list(map(float, cols[1:]))).reshape(3, 3)
            # elif "T_03" in name:
            #    t03 = np.array(list(map(float, cols[1:]))).reshape(3)
            if "P_rect_02" in name:
                P_rect_left = np.array(list(map(float, cols[1:]))).reshape(3, 4)
            elif "P_rect_03" in name:
                P_rect_right = np.array(list(map(float, cols[1:]))).reshape(3, 4)
            elif "R_rect_02" in name:
                R_rect_02 = np.array(list(map(float, cols[1:]))).reshape(3, 3)
            elif "R_rect_03" in name:
                R_rect_03 = np.array(list(map(float, cols[1:]))).reshape(3, 3)

    left_fx = P_rect_left[0, 0]
    left_fy = P_rect_left[1, 1]
    left_cx = P_rect_left[0, 2]
    left_cy = P_rect_left[1, 2]
    right_fx = P_rect_right[0, 0]
    right_fy = P_rect_right[1, 1]
    right_cx = P_rect_right[0, 2]
    right_cy = P_rect_right[1, 2]
    left_bx = P_rect_left[0, 3] / -left_fx
    right_bx = P_rect_right[0, 3] / -right_fx
    left_cam = CameraParameters(fx=left_fx, fy=left_fy, cx=left_cx, cy=left_cy)
    right_cam = CameraParameters(fx=right_fx, fy=right_fy, cx=right_cx, cy=right_cy)
    R_rect_23 = np.linalg.inv(R_rect_02) @ R_rect_03
    T23 = np.identity(4)
    # T23[:3, :3] = R_rect_23
    T23[0, 3] = right_bx - left_bx
    T02 = np.identity(4)
    T02[0, 3] = left_bx
    # T23[0, 3] = 0.03
    # zu weit rechts -> nach links schieben
    # pos nach links
    # neg nach rechts
    return StereoCamera(left_cam, right_cam, T23), T02


def get_gt_obj_segmentations_from_kitti(instance_file: str) -> List[ObjectDetection]:
    img = np.array(cv2.imread(instance_file, cv2.IMREAD_ANYDEPTH))
    obj_ids = np.unique(img)
    obj_detections = []
    for obj_id in obj_ids:
        if obj_id in [0, 10000]:
            continue
        track_id = obj_id % 1000
        obj_mask = img == obj_id
        convex_hull = cv2.convexHull(np.argwhere(obj_mask), returnPoints=True).reshape(
            -1, 2
        )
        convex_hull = np.flip(convex_hull)
        obj_mask = ObjectDetection(
            convex_hull=list(map(tuple, convex_hull.tolist())), track_id=track_id
        )
        if len(obj_mask.convex_hull) < 2:
            continue
        obj_detections.append(obj_mask)
    return obj_detections
