"""Contains preprocessing functionality, namely converting raw data to inputs for SLAM and MOT
"""
from typing import List

import numpy as np

from bamot.core.base_types import ObjectDetection, StereoCamera, StereoImage
from bamot.util.cv import get_convex_hull_mask


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
    # print(f"Transforming {len(left_object_pts)} points")
    for pt_2d in left_object_pts:
        pt_3d = stereo_camera.left.back_project(pt_2d).reshape(3, 1)
        pt_3d_hom = np.array([*pt_3d.tolist(), [1]]).reshape(4, 1)
        pt_3d_right_hom = (
            np.linalg.inv(stereo_camera.T_left_right) @ pt_3d_hom
        ).reshape(4, 1)
        pt_2d_right_hom = stereo_camera.right.project(pt_3d_right_hom).reshape(3, 1)
        pt_2d_right = (pt_2d_right_hom[:2] / pt_2d_right_hom[2]).reshape(2, 1)
        right_obj_pts.append(pt_2d_right)
        # print(pt_2d_right)
        # print(pt_2d)
        # print("Transformed single point")
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
        left_hull_pts = np.array(obj.convex_hull)
        left_obj_mask = get_convex_hull_mask(left_hull_pts, img_shape)
        left_mask[left_obj_mask] = 0
        right_obj_pts = transform_object_points(left_hull_pts, stereo_camera)
        # right_obj_pts = left_hull_pts.reshape(-1, 2)
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
    return StereoImage(masked_left_image_slam, masked_right_image_slam)
