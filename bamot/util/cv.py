import logging
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw

from bamot.core.base_types import (CameraParameters, Feature, FeatureMatcher,
                                   Landmark, Match)

LOGGER = logging.getLogger("UTIL:CV")


def project(params: CameraParameters, pt_3d: np.ndarray):
    x, y, z = map(float, pt_3d)
    return np.array(
        [params.fx * (x / z) + params.cx, params.fy * (y / z) + params.cy]
    ).reshape(2, 1)


def back_project(params: CameraParameters, pt_2d: np.ndarray):
    pt_2d = pt_2d.reshape(2, 1)
    u, v = map(float, pt_2d)
    mx = (u - params.cx) / params.fx
    my = (v - params.cy) / params.fy
    length = np.sqrt(mx ** 2 + my ** 2 + 1)
    return (np.array([mx, my, 1]) / length).reshape(3, 1)


def get_convex_hull(points_2d: np.ndarray):
    return cv2.convexHull(
        points_2d.reshape(-1, 2).astype(int), returnPoints=True
    ).reshape(-1, 2)


def get_convex_hull_mask(
    points_2d: np.ndarray, img_shape: Tuple[int, int]
) -> List[np.ndarray]:
    """Produces the convex hull mask given a list of 2d points.

    :param points_2d: a list of 2d points.
    :type points_2d: an np.ndarray
    :param img_shape: the shape of the image (height, width)
    :type img_shape: a tuple of two ints
    :returns: a numpy array of the given img shape where all points inside
    the convex hull of the points_2d are set to True, and all others to False
    :rtype: a np.ndarray

    """
    height, width = img_shape[:2]
    img = Image.new("L", (width, height), 0)
    hull = get_convex_hull(np.flip(points_2d))
    hull = [(p[1], p[0]) for p in hull]  # PIL expects different order of x,y
    ImageDraw.Draw(img).polygon(hull, outline=1, fill=1)
    mask = np.array(img)
    return mask == 1


def mask_img(
    mask: np.ndarray, img: np.ndarray, dilate: Union[bool, int] = False
) -> np.ndarray:
    for _ in range(int(dilate)):
        mask = cv2.dilate(mask.astype(float), kernel=np.ones((3, 3))).astype(int)
    masked_img = np.zeros(img.shape)
    masked_img[mask] = img[mask]
    return masked_img.astype(np.uint8)


def to_homogeneous_pt(pt: np.ndarray) -> np.ndarray:
    pt_hom = np.array([*pt.reshape(-1,), 1]).reshape(-1, 1)
    return pt_hom


def from_homogeneous_pt(pt_hom: np.ndarray) -> np.ndarray:
    pt = pt_hom[:-1] / pt_hom[-1]
    return pt


def get_orb_feature_matcher(num_features: int = 4000):
    orb = cv2.ORB_create(nfeatures=num_features)
    matcher = cv2.BFMatcher_create(normType=cv2.NORM_HAMMING, crossCheck=True)

    def detect_features(
        img: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> List[Feature]:
        keypoints, descriptors = orb.detectAndCompute(
            img, mask=255 * mask.astype(np.uint8)
        )
        features = []
        if keypoints:
            for keypoint, descriptor in zip(keypoints, descriptors):
                features.append(
                    Feature(u=keypoint.pt[0], v=keypoint.pt[1], descriptor=descriptor)
                )
        return features

    def match_features(first: List[Feature], second: List[Feature]) -> List[Match]:
        if not first or not second:
            return []
        first_descriptors = np.array(list(map(lambda x: x.descriptor, first)))
        second_descriptors = np.array(list(map(lambda x: x.descriptor, second)))
        matches = matcher.match(first_descriptors, second_descriptors)
        return [(match.queryIdx, match.trainIdx) for match in matches]

    return FeatureMatcher(
        detect_features=detect_features, match_features=match_features
    )


def project_landmarks(landmarks: List[Landmark]):
    # TODO for object associations
    pass


def triangulate(
    vec_left: np.ndarray,
    vec_right: np.ndarray,
    R_left_right: np.ndarray,
    t_left_right: np.ndarray,
) -> np.ndarray:
    vec_right_unrotated = R_left_right @ vec_right
    b = np.zeros((2, 1))
    b[0] = t_left_right.T @ vec_left
    b[1] = t_left_right.T @ vec_right_unrotated
    A = np.zeros((2, 2))
    A[0, 0] = vec_left.T @ vec_left
    A[1, 0] = vec_left.T @ vec_right_unrotated
    A[0, 1] = -A[1, 0]
    A[1, 1] = -vec_right_unrotated.T @ vec_right_unrotated
    l = np.linalg.inv(A) @ b
    xm = l[0] * vec_left
    xn = t_left_right + l[1] * vec_right_unrotated
    return ((xm + xn) / 2).reshape((3, 1))
