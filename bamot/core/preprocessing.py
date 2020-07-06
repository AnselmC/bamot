"""Contains preprocessing functionality, namely converting raw data to inputs for SLAM and MOT
"""
# pylint: disable=invalid-name
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from bamot.core.base_types import (Feature, ObjectDetection, StereoCamera,
                                   StereoImage)


def get_convex_hull_mask(
    points_2d: np.ndarray, img_shape: Tuple[int, int]
) -> List[np.ndarray]:
    """Produces the convex hull mask given a list of 2d points.

    :param points_2d: a list of 2d points.
    :type points_2d: an np.ndarray
    :param img_shape: the shape of the image (height, width)
    :type img_shape: a tuple of two ints
    :returns: a numpy array of the given img shape where all points inside the convex hull of the points_2d are    set to True, and all others to False
    :rtype: a np.ndarray

    """
    height, width = img_shape[:2]
    img = Image.new("L", (width, height), 0)
    points_2d = np.array(points_2d).reshape(-1, 2).astype(int)
    hull = cv2.convexHull(points_2d).flatten().reshape(-1, 2)
    hull = [(p[1], p[0]) for p in hull]
    mask = np.zeros(img_shape[:2])
    mask = cv2.fillConvexPoly(mask, np.int32(hull), (1, 1, 1))

    if len(points_2d) > 2:
        if isinstance(points_2d[0], Feature):
            points_2d_arr = []
            for pt in points_2d:
                points_2d_arr.append(np.array([pt.u, pt.v]))
            points_2d = points_2d_arr

        hull_idxs = (
            cv2.convexHull(np.array(points_2d).astype(int)).flatten().reshape(-1, 2)
        )
        hull_pts = [(p[1], p[0]) for p in hull_idxs]
        ImageDraw.Draw(img).polygon(hull_pts, outline=1, fill=1)
    mask = np.array(img)
    return mask == 1


def transform_object_points(
    left_object_pts: np.ndarray, stereo_camera: StereoCamera
) -> List[np.ndarray]:
    """Transforms points from left image into right image.

    :param left_object_pts: The 2d points given in left image coordinates
    :type left_object_pts: an array of n points with the shape (n, 2)
    :param stereo_camera: the stereo camera setup
    :type stereo_camera: a StereoCamera data structure
    :returns: the transformed 2d points
    :rtype: a list of np.ndarrays
    """
    right_obj_pts = []
    print(f"Transforming {len(left_object_pts)} points")
    for pt_2d in left_object_pts:
        pt_3d = stereo_camera.left.back_project(pt_2d).reshape(3, 1)
        pt_3d_hom = np.array([*pt_3d.tolist(), [1]]).reshape(4, 1)
        pt_3d_right_hom = (
            np.linalg.inv(stereo_camera.T_left_right) @ pt_3d_hom
        ).reshape(4, 1)
        pt_2d_right_hom = stereo_camera.right.project(pt_3d_right_hom).reshape(3, 1)
        pt_2d_right = (pt_2d_right_hom[:2] / pt_2d_right_hom[2]).reshape(2, 1)
        right_obj_pts.append(pt_2d_right)
        print(pt_2d_right)
        print(pt_2d)
        print("Transformed single point")
    return np.array(right_obj_pts).reshape(-1, 2).astype(int)


def preprocess_frame(
    stereo_image: StereoImage,
    stereo_camera: StereoCamera,
    object_detections: List[ObjectDetection],
) -> StereoImage:
    """Masks out object detections from a stereo image and returns the masked image.

    :param stereo_image: the raw stereo image data
    :type stereo_image: a StereoImage
    :param stereo_camera: the stereo camera setup
    :type stereo_camera: a StereoCamera
    :param object_detections: the object detections 
    :type object_detections: a list of ObjectDetections
    :returns: the masked stereo image
    :rtype: a StereoImage

    """
    raw_left_image, raw_right_image = stereo_image.left, stereo_image.right
    img_shape = raw_left_image.shape
    left_mask, right_mask = (np.ones(img_shape, dtype=np.uint8) for _ in range(2))

    for obj in object_detections:
        # get masks for object
        print("got left pts")
        left_mask[obj.object_mask] = 0
        left_obj_pts = np.argwhere(obj.object_mask).reshape(-1, 2)
        left_hull_pts = cv2.convexHull(
            np.array(left_obj_pts).astype(np.float32), returnPoints=True
        )
        right_obj_pts = transform_object_points(left_hull_pts, stereo_camera)
        # right_obj_pts = left_hull_pts.reshape(-1, 2)
        print(left_hull_pts.shape)
        print(right_obj_pts.shape)
        print("transformed pts")
        right_obj_mask = get_convex_hull_mask(right_obj_pts, img_shape)
        # right_obj_mask = obj.object_mask
        # print("got convex hull")
        ## add obj to mask
        right_mask[right_obj_mask] = 0

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
        StereoImage(masked_left_image_mot, masked_right_image_mot),
    )
