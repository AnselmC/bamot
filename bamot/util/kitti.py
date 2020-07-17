from functools import partial
from pathlib import Path
from typing import List

import cv2
import numpy as np

from bamot.core.base_types import (CameraParameters, ObjectDetection,
                                   StereoCamera)


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


def get_cameras_from_kitti(calib_file: Path) -> StereoCamera:
    with open(calib_file.as_posix(), "r") as fd:
        for line in fd:
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
    T23 = np.identity(4)
    T23[0, 3] = left_bx - right_bx
    # T23[0, 3] = 0.03
    # zu weit rechts -> nach links schieben
    # pos nach links
    # neg nach rechts
    # T02 = np.identity(4)
    # T02[:3, :3] = R02
    # T02[:3, 3] = t02
    # T03 = np.identity(4)
    # T03[:3, :3] = R03
    # T03[:3, 3] = t03
    # T23 = np.linalg.inv(T02) @ T03
    # left_cam = Camera(
    #    project=partial(_project, intrinsics=intrinsics_02),
    #    back_project=partial(_back_project, intrinsics=intrinsics_02),
    # )
    # right_cam = Camera(
    #    project=partial(_project, intrinsics=intrinsics_03),
    #    back_project=partial(_back_project, intrinsics=intrinsics_03),
    # )
    return StereoCamera(left_cam, right_cam, T23)


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
        obj_detections.append(obj_mask)
    return obj_detections
