"""Contains preprocessing functionality, namely converting raw data to inputs for SLAM and MOT
"""
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from bamot.core.base_types import (ObjectDetection, StereoCamera, StereoImage,
                                   StereoObjectDetection)
from bamot.util.cv import (dilate_mask, get_convex_hull_from_mask,
                           get_convex_hull_mask, get_feature_matcher)
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon


def _draw_contours(mask, img, color):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(img, contours, -1, color, 3)


def match_detections(
    left_object_detections, right_object_detections, stereo_image, only_iou
):
    num_left = len(left_object_detections)
    num_right = len(right_object_detections)
    if num_left >= num_right:
        num_first = num_left
        num_second = num_right
        left_first = True
        first_obj_detections = left_object_detections
        second_obj_detections = right_object_detections
        first_image = stereo_image.left
        second_image = stereo_image.right
    else:
        num_first = num_right
        num_second = num_left
        left_first = False
        first_obj_detections = right_object_detections
        second_obj_detections = left_object_detections
        first_image = stereo_image.right
        second_image = stereo_image.left

    cost_matrix = np.zeros((num_first, num_second))
    feature_matcher = get_feature_matcher()

    second_feature_map = {}
    for i, first_obj in enumerate(first_obj_detections):
        first_convex_hull = get_convex_hull_from_mask(first_obj.mask)
        first_features = feature_matcher.detect_features(first_image, first_obj.mask)
        first_obj.features = first_features
        if len(first_convex_hull) >= 3:
            first_obj_area = Polygon(get_convex_hull_from_mask(first_obj.mask))
        else:
            first_obj_area = Polygon()
        for j, second_obj in enumerate(second_obj_detections):
            if second_feature_map.get(j):
                second_features = second_feature_map[j]
            else:
                second_features = feature_matcher.detect_features(
                    second_image, second_obj.mask
                )
                second_feature_map[j] = second_features
            second_obj.features = second_features
            matched_features = feature_matcher.match_features(
                first_features, second_features
            )
            second_convex_hull = get_convex_hull_from_mask(second_obj.mask)
            if len(second_convex_hull) >= 3:
                second_obj_area = Polygon(get_convex_hull_from_mask(second_obj.mask))
            else:
                second_obj_area = Polygon()
            iou = (
                first_obj_area.intersection(second_obj_area).area
                / first_obj_area.union(second_obj_area).area
            )
            normalized_matched_features = len(matched_features) / max(
                1, min(len(first_features), len(second_features))
            )
            cost_matrix[i][j] = iou if only_iou else iou + normalized_matched_features

    first_indices, second_indices = linear_sum_assignment(cost_matrix, maximize=True)

    good_matches = []
    for row_idx, col_idx in zip(first_indices, second_indices):
        edge_weight = cost_matrix[row_idx, col_idx].sum()
        if edge_weight == 0:
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
    colors: Dict[int, Tuple[int, int, int]],
    left_object_detections: List[ObjectDetection],
    use_right_tracks: bool = False,
    right_object_detections: Optional[List[ObjectDetection]] = None,
    only_iou: bool = False,
    use_unmatched: bool = False,
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
            left_object_detections, right_object_detections, stereo_image, only_iou
        )
    else:
        matched_detections = []

    matched = set()
    feature_matcher = get_feature_matcher()
    if matched_detections:
        for left_obj_idx, right_obj_idx in matched_detections:
            left_obj = left_object_detections[left_obj_idx]
            right_obj = right_object_detections[right_obj_idx]
            left_mask[left_obj.mask] = 0
            right_mask[right_obj.mask] = 0
            if use_right_tracks:
                matched.add(right_obj_idx)
                left_obj.track_id = right_obj.track_id
            else:
                matched.add(left_obj_idx)
                right_obj.track_id = left_obj.track_id
            if colors:
                color = colors[left_obj.track_id]
                _draw_contours(left_obj.mask, raw_left_image, color)
                _draw_contours(right_obj.mask, raw_right_image, color)
                left_keypoints = [
                    cv2.KeyPoint(x=f.u, y=f.v, _size=1) for f in left_obj.features
                ]
                right_keypoints = [
                    cv2.KeyPoint(x=f.u, y=f.v, _size=1) for f in right_obj.features
                ]
                cv2.drawKeypoints(raw_left_image, left_keypoints, raw_left_image)
                cv2.drawKeypoints(raw_right_image, right_keypoints, raw_right_image)
            stereo_object_detections.append(StereoObjectDetection(left_obj, right_obj))
    if use_right_tracks:
        unmatched = set(range(len(right_object_detections))).difference(matched)
    else:
        unmatched = set(range(len(left_object_detections))).difference(matched)
    if unmatched:
        for idx in unmatched:
            if use_right_tracks:
                obj = right_object_detections[idx]
            else:
                obj = left_object_detections[idx]
            # get masks for object
            obj_mask = obj.mask
            hull_pts = np.array(get_convex_hull_from_mask(obj_mask))

            other_obj_mask = get_convex_hull_mask(np.flip(hull_pts), img_shape)
            other_obj_mask = dilate_mask(
                other_obj_mask, num_pixels=int(obj_mask.sum() * 0.01)
            )
            if use_right_tracks:
                right_mask[obj_mask] = 0
                if use_unmatched:
                    left_mask[other_obj_mask] = 0
                    features = feature_matcher.detect_features(
                        raw_left_image, other_obj_mask
                    )
                    left_obj = ObjectDetection(
                        mask=other_obj_mask,
                        track_id=obj.track_id,
                        cls=obj.cls,
                        features=features,
                    )
                    stereo_object_detections.append(
                        StereoObjectDetection(left_obj, obj)
                    )
                    if colors:
                        color = colors[obj.track_id]
                        _draw_contours(obj_mask, raw_right_image, color)
                        _draw_contours(other_obj_mask, raw_left_image, color)
                        left_keypoints = [
                            cv2.KeyPoint(x=f.u, y=f.v, _size=1)
                            for f in left_obj.features
                        ]
                        right_keypoints = [
                            cv2.KeyPoint(x=f.u, y=f.v, _size=1) for f in obj.features
                        ]
                        cv2.drawKeypoints(
                            raw_left_image, left_keypoints, raw_left_image
                        )
                        cv2.drawKeypoints(
                            raw_right_image, right_keypoints, raw_right_image
                        )
            else:
                left_mask[obj_mask] = 0
                if use_unmatched:
                    right_mask[other_obj_mask] = 0
                    features = feature_matcher.detect_features(
                        raw_right_image, other_obj_mask
                    )
                    right_obj = ObjectDetection(
                        mask=other_obj_mask,
                        track_id=obj.track_id,
                        cls=obj.cls,
                        features=features,
                    )
                    stereo_object_detections.append(
                        StereoObjectDetection(obj, right_obj)
                    )
                    if colors:
                        color = colors[obj.track_id]
                        _draw_contours(obj_mask, raw_left_image, color)
                        _draw_contours(other_obj_mask, raw_right_image, color)
                        left_keypoints = [
                            cv2.KeyPoint(x=f.u, y=f.v, _size=1) for f in obj.features
                        ]
                        right_keypoints = [
                            cv2.KeyPoint(x=f.u, y=f.v, _size=1)
                            for f in right_obj.features
                        ]
                        cv2.drawKeypoints(
                            raw_left_image, left_keypoints, raw_left_image
                        )
                        cv2.drawKeypoints(
                            raw_right_image, right_keypoints, raw_right_image
                        )

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
