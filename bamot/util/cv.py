from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw

from bamot.core.base_types import Feature, FeatureMatcher, Landmark, Match


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
    hull = get_convex_hull(points_2d)
    hull = [(p[1], p[0]) for p in hull]  # PIL expects different order of x,y
    ImageDraw.Draw(img).polygon(hull, outline=1, fill=1)
    mask = np.array(img)
    return mask == 1


def mask_img(
    mask: np.ndarray, img: np.ndarray, dilate: Union[bool, int] = False
) -> np.ndarray:
    for _ in range(int(dilate)):
        mask = cv2.dilate(mask, kernel=np.ones((3, 3)))
    masked_img = np.zeros(img.shape)
    masked_img[mask] = img[mask]
    return masked_img


def get_orb_feature_matcher(num_features: int = 100):
    orb = cv2.ORB_create(nfeatures=num_features)
    matcher = cv2.BFMatcher_create(normType=cv2.NORM_HAMMING)

    def detect_features(
        img: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> List[Feature]:
        keypoints, descriptors = orb.detectAndCompute(img, mask=mask)
        features = []
        if keypoints:
            for keypoint, descriptor in zip(keypoints, descriptors):
                features.append(
                    Feature(u=keypoint.pt[0], v=keypoint.pt[1], descriptor=descriptor)
                )
        return features

    def match_features(first: List[Feature], second: List[Feature]) -> List[Match]:
        first_descriptors = np.array(list(map(lambda x: x.descriptor, first)))
        second_descriptors = np.array(list(map(lambda x: x.descriptor, second)))
        matches = matcher.match(first_descriptors, second_descriptors)
        return [Match((match.queryIdx, match.trainIdx)) for match in matches]

    return FeatureMatcher(
        detect_features=detect_features, match_features=match_features
    )


def project_landmarks(landmarks: List[Landmark]):
    pass
