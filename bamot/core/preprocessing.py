"""Contains preprocessing functionality, namely converting raw data to inputs for SLAM and MOT
"""
from typing import List, Optional, Tuple

import numpy as np
from bamot.core.base_types import (ObjectDetection, StereoCamera, StereoImage,
                                   StereoObjectDetection)
from bamot.util.cv import (back_project, dilate_mask, from_homogeneous_pt,
                           get_convex_hull_from_mask, get_convex_hull_mask,
                           project, to_homogeneous_pt)
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon


def transform_object_points(
    left_object_pts: np.ndarray, stereo_camera: StereoCamera
) -> np.ndarray:
    """Transforms points from left image into right image.

    :param left_object_pts: The 2d points given in left image coordinates
    :type left_object_pts: an array of n points with the shape (n, 2)
    :param stereo_camera: the stereo camera setup
    :type stereo_camera: a StereoCamera data structure
    :returns: the transformed 2d points
    :rtype: an np.ndarray of shape (num_points, 2)
    """
    right_obj_pts = []
    for pt_2d in left_object_pts:
        pt_3d = back_project(stereo_camera.left, pt_2d).reshape(3, 1)
        pt_3d_hom = to_homogeneous_pt(pt_3d)
        pt_3d_right_hom = (
            np.linalg.inv(stereo_camera.T_left_right) @ pt_3d_hom
        ).reshape(4, 1)
        pt_2d_right = project(
            stereo_camera.right, from_homogeneous_pt(pt_3d_right_hom)
        ).reshape(2, 1)
        right_obj_pts.append(pt_2d_right)
    return np.array(right_obj_pts).reshape(-1, 2).astype(int)


def match_detections(left_object_detections, right_object_detections):
    num_left = len(left_object_detections)
    num_right = len(right_object_detections)
    if num_left >= num_right:
        num_first = num_left
        num_second = num_right
        left_first = True
        first_obj_detections = left_object_detections
        second_obj_detections = right_object_detections
    else:
        num_first = num_right
        num_second = num_left
        left_first = False
        first_obj_detections = right_object_detections
        second_obj_detections = left_object_detections

    cost_matrix = np.zeros((num_first, num_second))
    for i, first_obj in enumerate(first_obj_detections):
        first_convex_hull = get_convex_hull_from_mask(first_obj.mask)
        if len(first_convex_hull) >= 3:
            first_obj_area = Polygon(get_convex_hull_from_mask(first_obj.mask))
        else:
            first_obj_area = Polygon()
        for j, second_obj in enumerate(second_obj_detections):
            second_convex_hull = get_convex_hull_from_mask(second_obj.mask)
            if len(second_convex_hull) >= 3:
                second_obj_area = Polygon(get_convex_hull_from_mask(second_obj.mask))
            else:
                second_obj_area = Polygon()
            iou = (
                first_obj_area.intersection(second_obj_area).area
                / first_obj_area.union(second_obj_area).area
            )
            cost_matrix[i][j] = iou

    first_indices, second_indices = linear_sum_assignment(cost_matrix, maximize=True)

    good_matches = []
    for row_idx, col_idx in zip(first_indices, second_indices):
        iou = cost_matrix[row_idx, col_idx].sum()
        if iou == 0:
            continue

        if left_first:
            left_idx = row_idx
            right_idx = col_idx
        else:
            right_idx = row_idx
            left_idx = col_idx
        good_matches.append((left_idx, right_idx))

    return good_matches


def preprocess_frame(
    stereo_image: StereoImage,
    stereo_camera: StereoCamera,
    left_object_detections: List[ObjectDetection],
    right_object_detections: Optional[List[ObjectDetection]] = None,
) -> Tuple[StereoImage, List[StereoObjectDetection]]:
    """Masks out object detections from a stereo image and returns the masked image.

    :param stereo_image: the raw stereo image data
    :type stereo_image: a StereoImage
    :param stereo_camera: the stereo camera setup
    :type stereo_camera: a StereoCamera
    :param object_detections: the object detections 
    :type object_detections: a list of ObjectDetections
    :returns: the masked stereo image and a list of StereoObjectDetections
    :rtype: a StereoImage, a list of StereoObjectDetection

    """
    raw_left_image, raw_right_image = stereo_image.left, stereo_image.right
    img_shape = raw_left_image.shape
    left_mask, right_mask = (np.ones(img_shape, dtype=np.uint8) for _ in range(2))

    stereo_object_detections = []
    if left_object_detections and right_object_detections:
        matched_detections = match_detections(
            left_object_detections, right_object_detections
        )
    else:
        matched_detections = []

    matched_left = set()
    if matched_detections:
        for left_obj_idx, right_obj_idx in matched_detections:
            matched_left.add(left_obj_idx)
            left_obj = left_object_detections[left_obj_idx]
            right_obj = right_object_detections[right_obj_idx]
            left_mask[left_obj.mask] = 0
            right_mask[right_obj.mask] = 0
            right_obj.track_id = left_obj.track_id
            stereo_object_detections.append(StereoObjectDetection(left_obj, right_obj))
    unmatched_left = set(range(len(left_object_detections))).difference(matched_left)
    if unmatched_left:
        for left_idx in unmatched_left:
            obj = left_object_detections[left_idx]
            # get masks for object
            left_obj_mask = obj.mask
            left_mask[left_obj_mask] = 0
            left_hull_pts = np.array(get_convex_hull_from_mask(left_obj_mask))

            right_obj_pts = transform_object_points(left_hull_pts, stereo_camera)
            right_obj_mask = get_convex_hull_mask(np.flip(right_obj_pts), img_shape)
            right_obj_mask = dilate_mask(right_obj_mask, num_pixels=5)
            right_mask[right_obj_mask] = 0
            right_obj = ObjectDetection(
                mask=right_obj_mask, track_id=obj.track_id, cls=obj.cls,
            )
            stereo_object_detections.append(StereoObjectDetection(obj, right_obj))

    left_mask = left_mask == 0
    right_mask = right_mask == 0
    masked_left_image_slam = raw_left_image.copy()
    masked_right_image_slam = raw_right_image.copy()
    masked_left_image_slam[left_mask] = 0
    masked_right_image_slam[right_mask] = 0
    masked_left_image_mot = np.zeros(img_shape, dtype=np.uint8)
    masked_right_image_mot = np.zeros(img_shape, dtype=np.uint8)
    masked_left_image_mot[left_mask] = raw_left_image[left_mask]
    masked_right_image_mot[right_mask] = raw_right_image[right_mask]
    return (
        StereoImage(masked_left_image_slam, masked_right_image_slam),
        stereo_object_detections,
    )
