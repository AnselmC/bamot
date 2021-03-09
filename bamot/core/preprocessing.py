"""Contains preprocessing functionality, namely converting raw data to inputs for SLAM and MOT
"""
import uuid
from typing import List, Optional, Tuple

import cv2
import numpy as np
from bamot.core.base_types import (ObjectDetection, StereoImage,
                                   StereoObjectDetection)
from bamot.util.cv import (dilate_mask, draw_contours,
                           get_convex_hull_from_mask, get_feature_matcher)
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon


def draw_contours_and_text(
    obj: ObjectDetection,
    other_obj: ObjectDetection,
    img: np.ndarray,
    other_img: np.ndarray,
    color: np.ndarray,
):
    draw_contours(obj.mask, img, color, 3)
    draw_contours(other_obj.mask, other_img, color, 3)
    y, x = map(min, np.where(obj.mask != 0))
    img = cv2.putText(
        img, str(obj.track_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3,
    )
    y, x = map(min, np.where(other_obj.mask != 0))
    other_img = cv2.putText(
        other_img, str(obj.track_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3,
    )
    if obj.features:
        keypoints = [cv2.KeyPoint(x=f.u, y=f.v, _size=1) for f in obj.features]
        cv2.drawKeypoints(img, keypoints, img)
    if other_obj.features:
        other_keypoints = [
            cv2.KeyPoint(x=f.u, y=f.v, _size=1) for f in other_obj.features
        ]
        cv2.drawKeypoints(other_img, other_keypoints, other_img)


def transform_unmatched_to_other_mask(
    unmatched,
    detections,
    mask,
    other_mask,
    img,
    other_img,
    track_ids,
    stereo_object_detections,
    img_shape,
    colors,
):
    feature_matcher = get_feature_matcher()
    for unmatched_idx in unmatched:
        obj = detections[unmatched_idx]
        obj_mask = obj.mask
        other_obj_mask = dilate_mask(
            obj_mask, num_pixels=min(10, max(1, 1000 // int(obj_mask.sum()))),
        )
        features = feature_matcher.detect_features(other_img, other_obj_mask)
        other_obj_mask[other_mask == 0] = 0
        if not other_obj_mask.sum() or not features:
            continue
        other_mask[other_obj_mask] = 0
        mask[obj_mask] = 0
        track_id = uuid.uuid1().int if obj.track_id in track_ids else obj.track_id
        obj.track_id = track_id

        other_obj = ObjectDetection(
            mask=other_obj_mask, track_id=track_id, cls=obj.cls, features=features,
        )
        color = colors[track_id]
        draw_contours_and_text(obj, other_obj, img, other_img, color)
        stereo_object_detections.append(StereoObjectDetection(obj, other_obj))
        track_ids.add(track_id)


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
            if first_obj.cls != second_obj.cls:
                continue
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
    left_object_detections: List[ObjectDetection],
    colors,
    right_object_detections: Optional[List[ObjectDetection]] = None,
    only_iou: bool = False,
    use_unmatched: bool = False,
) -> Tuple[StereoImage, List[StereoObjectDetection]]:
    """Masks out object detections from a stereo image and returns the masked image.

    :param stereo_image: the raw stereo image data
    :type stereo_image: a StereoImage
    :param object_detections: the object detections 
    :type object_detections: a list of ObjectDetections
    :returns: the masked stereo image and a list of StereoObjectDetections
    :rtype: a StereoImage, a list of StereoObjectDetection

    """
    raw_left_image, raw_right_image = stereo_image.left, stereo_image.right
    img_shape = raw_left_image.shape
    left_mask, right_mask = (np.ones(img_shape[:2], dtype=np.uint8) for _ in range(2))

    stereo_object_detections = []
    if left_object_detections and right_object_detections:
        matched_detections = match_detections(
            left_object_detections, right_object_detections, stereo_image, only_iou
        )
    else:
        matched_detections = []

    left_matched = set()
    right_matched = set()
    track_ids = set()
    for left_obj_idx, right_obj_idx in matched_detections:
        left_obj = left_object_detections[left_obj_idx]
        right_obj = right_object_detections[right_obj_idx]
        left_mask[left_obj.mask] = 0
        right_mask[right_obj.mask] = 0
        if left_obj.track_id in track_ids:
            if right_obj.track_id in track_ids:
                track_id = uuid.uuid1().int
            else:
                track_id = right_obj.track_id
        else:
            track_id = left_obj.track_id
        left_obj.track_id = track_id
        right_obj.track_id = track_id
        color = colors[track_id]
        draw_contours_and_text(
            left_obj, right_obj, raw_left_image, raw_right_image, color
        )
        stereo_object_detections.append(StereoObjectDetection(left_obj, right_obj))
        left_matched.add(left_obj_idx)
        right_matched.add(right_obj_idx)
        track_ids.add(track_id)
    if use_unmatched:
        right_unmatched = set(range(len(right_object_detections))).difference(
            right_matched
        )
        left_unmatched = set(range(len(left_object_detections))).difference(
            left_matched
        )
        transform_unmatched_to_other_mask(
            unmatched=left_unmatched,
            detections=left_object_detections,
            mask=left_mask,
            other_mask=right_mask,
            img=raw_left_image,
            other_img=raw_right_image,
            track_ids=track_ids,
            stereo_object_detections=stereo_object_detections,
            img_shape=img_shape,
            colors=colors,
        )
        transform_unmatched_to_other_mask(
            unmatched=right_unmatched,
            detections=right_object_detections,
            mask=right_mask,
            other_mask=left_mask,
            img=raw_right_image,
            other_img=raw_left_image,
            track_ids=track_ids,
            stereo_object_detections=stereo_object_detections,
            img_shape=img_shape,
            colors=colors,
        )

    left_mask = left_mask == 0
    right_mask = right_mask == 0
    masked_left_image_slam = raw_left_image.copy()
    masked_right_image_slam = raw_right_image.copy()
    masked_left_image_slam[left_mask] = 0
    masked_right_image_slam[right_mask] = 0
    masked_left_image_mot = 255 * np.ones(img_shape, dtype=np.uint8)
    masked_right_image_mot = 255 * np.ones(img_shape, dtype=np.uint8)
    masked_left_image_mot[left_mask] = raw_left_image[left_mask]
    masked_right_image_mot[right_mask] = raw_right_image[right_mask]
    return (
        StereoImage(
            masked_left_image_slam,
            masked_right_image_slam,
            img_width=stereo_image.img_width,
            img_height=stereo_image.img_height,
        ),
        stereo_object_detections,
    )
