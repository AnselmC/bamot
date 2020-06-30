"""Contains preprocessing functionality, namely converting raw data to inputs for SLAM and MOT
"""
# pylint: disable=invalid-name
from typing import Callable, Dict, List, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw

from bamot.core.base_types import (Camera, ExtendedObjectDetection3D, Feature,
                                   ObjectDetection3D)


def get_object_detections(
    stereo_images: Tuple[np.ndarray, np.ndarray],
    detect_objects_fn: Callable[
        [Tuple[np.ndarray, np.ndarray]], List[Dict[str, float]]
    ],
) -> List[ObjectDetection3D]:
    """Returns list of ObjectDetection3D given stereo images and a detector fn.

    :param stereo_images: The stereo images to detect 3D objects for
    :type stereo_images: a two-tuple (left, right) of np.ndarrays (images)
    :param detect_objects_fn: an object detection callable
    :type detect_objects_fn: a callable that takes to np.ndarrays and returns
                             a list of dict objects which hold the keys: x, y,
                             z, height, width, length
    :returns: a list of 3d object detections
    """
    object_detections = detect_objects_fn(stereo_images)
    return [ObjectDetection3D(**detection) for detection in object_detections]


def get_feature_detections(
    image: np.ndarray,
    detect_features_fn: Callable[[np.ndarray], List[Dict[str, Union[int, np.ndarray]]]],
) -> List[Feature]:
    features = detect_features_fn(image)
    return [Feature(**feature) for feature in features]


def get_convex_hull_mask(
    points_2d: List[Union[np.ndarray, Feature]], img_shape: Tuple[int, int]
) -> List[np.ndarray]:
    height, width = img_shape[:2]
    img = Image.new("L", (width, height), 0)
    if len(points_2d) > 2:
        if isinstance(points_2d[0], Feature):
            points_2d_arr = []
            for pt in points_2d:
                points_2d_arr.append(np.array([pt.u, pt.v]))
            points_2d = points_2d_arr

        hull_pts = cv2.convexHull(
            np.array(points_2d).astype(np.float32), returnPoints=True
        )
        ImageDraw.Draw(img).polygon(hull_pts, outline=1, fill=1)
    mask = np.array(img)
    return mask == 1


def to_homogeneous(pt: np.ndarray) -> np.ndarray:
    pt = pt.reshape(3, 1)
    return cv2.convertPointsToHomogeneous(pt.T)[0].T


def from_homogeneous(pt: np.ndarray) -> np.ndarray:
    return cv2.convertPointsFromHomogeneous(pt.T)[0].T


def project_bounding_box(
    obj: ObjectDetection3D, cam: Camera, img_shape: Tuple[int, int]
) -> Tuple[np.ndarray, List[np.ndarray]]:
    # compute extreme points (8 corners of box)
    points = compute_extreme_points(obj)

    points_2d = []
    for point in points:
        # transform point into camera frame
        point_cam_hom = cam.T_0_i @ point
        # project point
        point_2d_hom = cam.project(point_cam_hom)
        point_2d = point_2d_hom[:2] / point_2d_hom[2]
        points_2d.append(point_2d)
    return get_convex_hull_mask(points_2d, img_shape), points_2d


def compute_extreme_points(obj: ObjectDetection3D) -> np.ndarray:
    x_corners = [
        -obj.length / 2,
        obj.length / 2,
        obj.length / 2,
        obj.length / 2,
        obj.length / 2,
        -obj.length / 2,
        -obj.length / 2,
        -obj.length / 2,
    ]
    y_corners = [-obj.height, -obj.height, 0, 0, -obj.height, -obj.height, 0, 0]
    z_corners = [
        -obj.width / 2,
        -obj.width / 2,
        -obj.width / 2,
        obj.width / 2,
        obj.width / 2,
        obj.width / 2,
        obj.width / 2,
        -obj.width / 2,
    ]
    corners = np.array([x_corners, y_corners, z_corners])
    obj_rot = np.array(
        [
            [np.cos(obj.ry), 0, np.sin(obj.ry)],
            [0, 1, 0],
            [-np.sin(obj.ry), 0, np.cos(obj.ry)],
        ]
    )
    center = np.array([obj.x, obj.y, obj.z]).reshape(3, 1)
    corners = cv2.convertPointsToHomogeneous((obj_rot @ corners + center).T).reshape(
        8, 4
    )

    # dimensions = np.array([obj.width, obj.height, obj.length]).reshape(3, 1)
    # center_pt_vec = np.repeat(center_pt, 8, axis=0)
    # dimensions_vec = np.repeat(dimensions, 8, axis=0)
    # weight_mat = np.diag(
    #    [
    #        -1,
    #        -1,
    #        -1,
    #        1,
    #        -1,
    #        1,
    #        1,
    #        1,
    #        -1,
    #        -1,
    #        1,
    #        -1,
    #        -1,
    #        -1,
    #        1,
    #        1,
    #        -1,
    #        1,
    #        1,
    #        1,
    #        1,
    #        -1,
    #        1,
    #        1,
    #    ]
    # )
    # extreme_pts = center_pt_vec + 0.5 * (weight_mat @ dimensions_vec)
    return corners


def preprocess_frame(
    stereo_images: Tuple[np.ndarray, np.ndarray],
    stereo_cameras: Tuple[Camera, Camera],
    detect_objects_fn: Callable[
        [Tuple[np.ndarray, np.ndarray]], List[Dict[str, float]]
    ],
    detect_features_fn: Callable[[np.ndarray], List[Dict[str, Union[int, np.ndarray]]]],
    use_feature_convex_hull: bool = False,
) -> Tuple[List[ExtendedObjectDetection3D], Tuple[np.ndarray, np.ndarray]]:
    raw_left_image, raw_right_image = stereo_images
    img_shape = raw_left_image.shape
    left_mask, right_mask = (np.ones(img_shape, dtype=np.uint8) for _ in range(2))
    left_cam, right_cam = stereo_cameras
    extended_objects = []
    objects = get_object_detections(stereo_images, detect_objects_fn)

    for obj in objects:
        # project 3D bounding box onto cams
        left_obj_mask, left_bounding_points = project_bounding_box(
            obj, left_cam, img_shape
        )
        right_obj_mask, right_bounding_points = project_bounding_box(
            obj, right_cam, img_shape
        )
        # mask out surrounding area
        left_obj, right_obj = (np.zeros(img_shape, dtype=np.uint8) for _ in range(2))
        left_obj[left_obj_mask] = raw_left_image[left_obj_mask]
        right_obj[right_obj_mask] = raw_right_image[right_obj_mask]
        # detect features on masked image
        left_features = get_feature_detections(left_obj, detect_features_fn)
        right_features = get_feature_detections(right_obj, detect_features_fn)
        if use_feature_convex_hull:
            left_feature_mask = get_convex_hull_mask(left_features, img_shape)
            right_feature_mask = get_convex_hull_mask(right_features, img_shape)
        else:
            left_feature_mask = left_obj_mask
            right_feature_mask = right_obj_mask
        # add feature masks
        left_mask[left_feature_mask] = 0
        right_mask[right_feature_mask] = 0
        extended_objects.append(
            ExtendedObjectDetection3D(
                object_detection=obj,
                left_projected_bounding_box=left_bounding_points,
                right_projected_bounding_box=right_bounding_points,
                left_features=left_features,
                right_features=right_features,
            )
        )

    left_mask = left_mask == 1
    right_mask = right_mask == 1
    masked_left_image = raw_left_image.copy()
    masked_right_image = raw_right_image.copy()
    masked_left_image[left_mask] = 0
    masked_right_image[right_mask] = 0
    return (extended_objects, (masked_left_image, masked_right_image))
