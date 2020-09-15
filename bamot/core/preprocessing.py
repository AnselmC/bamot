"""Contains preprocessing functionality, namely converting raw data to inputs for SLAM and MOT
"""
from typing import List, Tuple

import numpy as np
from bamot.core.base_types import (ObjectDetection, StereoCamera, StereoImage,
                                   StereoObjectDetection)
from bamot.util.cv import (back_project, from_homogeneous_pt,
                           get_convex_hull_mask, project, to_homogeneous_pt)


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


def preprocess_frame(
    stereo_image: StereoImage,
    stereo_camera: StereoCamera,
    object_detections: List[ObjectDetection],
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
    for obj in object_detections:
        # get masks for object
        left_hull_pts = np.array(obj.convex_hull)
        left_obj_mask = get_convex_hull_mask(left_hull_pts, img_shape)
        left_mask[left_obj_mask] = 0
        right_obj_pts = transform_object_points(left_hull_pts, stereo_camera)
        right_obj_mask = get_convex_hull_mask(right_obj_pts, img_shape)
        right_mask[right_obj_mask] = 0
        right_obj = ObjectDetection(
            convex_hull=list(map(tuple, right_obj_pts.tolist())), track_id=obj.track_id
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
