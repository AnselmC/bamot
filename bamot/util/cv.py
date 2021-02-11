import logging
import pickle
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from bamot.config import CONFIG as config
from bamot.core.base_types import (CameraParameters, Feature, FeatureMatcher,
                                   Landmark, Match, StereoCamera)
from g2o import AngleAxis
from PIL import Image, ImageDraw

if TYPE_CHECKING:
    import tensorflow as tf

if config.FEATURE_MATCHER != "orb":
    import bamot.thirdparty.SuperPoint.superpoint.match_features_demo as sp
    import tensorflow as tf

    tf.get_logger().setLevel(logging.ERROR)  # surpress TF1 -> TF2 warnings
    tf.config.threading.set_inter_op_parallelism_threads(
        2
    )  # s.t. extraction can run in parallel
    if tf.config.list_physical_devices("GPU"):
        LOADED = tf.saved_model.load(config.SUPERPOINT_WEIGHTS_PATH)
        MODEL = LOADED.signatures["serving_default"]

LOGGER = logging.getLogger("UTIL:CV")


class TriangulationError(Exception):
    pass


def get_corners_from_vector(vec: np.ndarray) -> np.ndarray:
    x, y, z, theta, height, width, length = vec.reshape(7, 1)
    translation = np.array([x, y, z]).reshape(3, 1)
    x_corners = length * [
        0.5,
        0.5,
        -0.5,
        -0.5,
        0.5,
        0.5,
        -0.5,
        -0.5,
    ]
    y_corners = height * [0, 0, 0, 0, -1, -1, -1, -1]
    z_corners = width * [
        0.5,
        -0.5,
        -0.5,
        0.5,
        0.5,
        -0.5,
        -0.5,
        0.5,
    ]
    corners = np.array([x_corners, y_corners, z_corners])
    rot = AngleAxis(theta, np.array([0, 1, 0])).rotation_matrix()
    return (rot @ corners) + translation


def get_convex_hull_from_mask(obj_mask):
    return list(map(tuple, get_convex_hull(np.argwhere(obj_mask))))


def project(params: CameraParameters, pt_3d_cam: np.ndarray):
    x, y, z = map(float, pt_3d_cam)
    return np.array(
        [params.fx * (x / z) + params.cx, params.fy * (y / z) + params.cy]
    ).reshape(2, 1)


def get_center_of_landmarks(landmarks: List[Landmark], reduction: str = "mean"):
    center = np.array([0.0, 0.0, 0.0]).reshape(3, 1)
    if not landmarks:
        return center
    if reduction == "mean":
        return np.mean([lm.pt_3d for lm in landmarks], axis=0)
    elif reduction == "median":
        return np.median([lm.pt_3d for lm in landmarks], axis=0)
    else:
        raise ValueError("Unknown reduction: %s", reduction)


def fullfills_epipolar_constraint(
    vec_left: np.ndarray,
    vec_right: np.ndarray,
    T_left_right: np.ndarray,
    threshold: float,
) -> bool:
    return (
        np.abs(vec_left.T @ compute_essential_matrix(T_left_right) @ vec_right)
        < threshold
    )


def compute_essential_matrix(T_left_right: np.ndarray) -> np.ndarray:
    t_left_right = T_left_right[:3, 3].reshape(3, 1)
    R_left_right = T_left_right[:3, :3]
    return get_skew_symmetric_matrix(t_left_right) @ R_left_right


def get_skew_symmetric_matrix(t_vec: np.ndarray) -> np.ndarray:
    if np.linalg.norm(t_vec) != 1:
        t_vec = t_vec / np.linalg.norm(t_vec)
    return np.array(
        [[0, -t_vec[2], t_vec[1]], [t_vec[2], 0, -t_vec[0]], [-t_vec[1], t_vec[0], 0]]
    )


def back_project(params: CameraParameters, pt_2d: np.ndarray):
    pt_2d = pt_2d.reshape(2, 1)
    u, v = map(float, pt_2d)
    mx = (u - params.cx) / params.fx
    my = (v - params.cy) / params.fy
    length = np.sqrt(mx ** 2 + my ** 2 + 1)
    return (np.array([mx, my, 1]) / length).reshape(3, 1)


def get_convex_hull(points_2d: np.ndarray):
    if len(points_2d):
        return np.flip(
            cv2.convexHull(
                points_2d.reshape(-1, 2).astype(int), returnPoints=True
            ).reshape(-1, 2)
        )
    return points_2d


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


def dilate_mask(mask: np.ndarray, num_pixels: int) -> np.ndarray:
    for _ in range(num_pixels):
        mask = cv2.dilate(mask.astype(float), kernel=np.ones((3, 3))).astype(bool)
    return mask


def mask_img(
    mask: np.ndarray, img: np.ndarray, dilate: Union[bool, int] = False
) -> np.ndarray:
    for _ in range(int(dilate)):
        mask = cv2.dilate(mask.astype(float), kernel=np.ones((3, 3))).astype(bool)
    masked_img = np.zeros(img.shape)
    masked_img[mask] = img[mask]
    return masked_img.astype(np.uint8)


def to_homogeneous(arr: np.ndarray) -> np.ndarray:
    if len(arr) != 3:
        raise RuntimeError(f"Array must be 3D to convert to 4D, but is {len(arr)}D")
    return np.vstack([arr, np.ones(len(arr.T)).reshape(1, -1)])


def from_homogeneous(arr: np.ndarray) -> np.ndarray:
    if len(arr) != 4:
        raise RuntimeError(f"Array must be 4D to convert to 3D, but is {len(arr)}D")
    return arr[:-1, :] / arr[-1, :]


def get_preprocessed_superpoint_feature_matcher(path: str):
    def detect_features(
        img: np.ndarray,
        mask: Optional[np.ndarray] = None,
        img_id: Optional[int] = None,
        track_index: Optional[int] = None,
        cam: Optional[str] = None,
    ) -> List[Feature]:
        feature_path = (
            Path(path)
            / f"{str(img_id).zfill(6)}"
            / f"{str(track_index).zfill(4)}"
            / f"{cam}.pkl"
        )
        with open(feature_path.as_posix(), "rb") as fp:
            features = pickle.load(fp)
        return features

    return FeatureMatcher(
        "SuperPoint",
        detect_features=detect_features,
        match_features=partial(match_features, norm=cv2.NORM_L2, threshold=2.0),
    )


def get_superpoint_feature_matcher():
    # TODO: investigate why this needs to be done
    import tensorflow as tf

    if tf.config.list_physical_devices("GPU"):
        global MODEL
    else:
        LOADED = tf.saved_model.load(config.SUPERPOINT_WEIGHTS_PATH)
        MODEL = LOADED.signatures["serving_default"]

    def preprocess_image(
        img: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> tf.Tensor:
        """
        Preprocesses a grayscale image for model forward pass.
        :param img: An rgb or grayscale image given as a numpy.ndarray (H, W)
        :returns: The image as a normalized 4D tensorflow Tensor
                  object (B, H, W, D) where B=1 and D=1.
        :raise: ValueError if img is
        """
        if len(img.shape) != 2:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if mask is not None:
            img = cv2.bitwise_and(img, mask.astype(np.uint8) * 255)
        if img.max() > 1:
            # normalize image
            img = img / 255.0
        img = img.astype(np.float32)
        # reshape image to be 4D
        img = np.expand_dims(img, 2)  # channel
        img = np.expand_dims(img, 0)  # batch
        return tf.constant(img)

    def detect_features(
        img: np.ndarray,
        mask: Optional[np.ndarray] = None,
        img_id: Optional[int] = None,
        track_index: Optional[int] = None,
        cam: Optional[str] = None,
    ) -> List[Feature]:
        # mask = None
        tensor = preprocess_image(img, mask)
        out = MODEL(tensor)
        kp_map = out["prob_nms"].numpy()[0].astype(np.float)
        desc_map = out["descriptors"].numpy()[0].astype(np.float)
        kp, desc = sp.extract_superpoint_keypoints_and_descriptors(
            kp_map, desc_map, config.NUM_FEATURES
        )
        desc = desc.astype(np.float32)
        return _get_features_from_kp_and_desc(kp, desc)

    return FeatureMatcher(
        "SuperPoint",
        detect_features=detect_features,
        match_features=partial(match_features, norm=cv2.NORM_L2, threshold=2.0),
    )


def _get_features_from_kp_and_desc(
    keypoints: List[cv2.KeyPoint], descriptors: np.ndarray
) -> List[Feature]:
    features = []
    if keypoints:
        for keypoint, descriptor in zip(keypoints, descriptors):
            features.append(
                Feature(u=keypoint.pt[0], v=keypoint.pt[1], descriptor=descriptor)
            )
    return features


def get_feature_matcher():
    name = config.FEATURE_MATCHER
    if name.lower() == "orb":
        return get_orb_feature_matcher()
    elif name.lower() in ["sp", "superpoint", "superpoints"]:
        return get_superpoint_feature_matcher()
    elif name.lower() == "superpoint_preprocessed":
        return get_preprocessed_superpoint_feature_matcher(
            path=config.SUPERPOINT_PREPROCESSED_PATH
        )
    else:
        raise ValueError(f"Unknown feature matcher: {name}")


def get_orb_feature_matcher():
    def detect_features(
        img: np.ndarray,
        mask: Optional[np.ndarray] = None,
        img_id: Optional[int] = None,
        track_index: Optional[int] = None,
        cam: Optional[str] = None,
    ) -> List[Feature]:
        # mask = None
        orb = cv2.ORB_create(nfeatures=config.NUM_FEATURES)
        keypoints, descriptors = orb.detectAndCompute(
            img, mask=255 * mask.astype(np.uint8) if mask is not None else None
        )
        return _get_features_from_kp_and_desc(keypoints, descriptors)

    return FeatureMatcher(
        "ORB",
        detect_features=detect_features,
        match_features=partial(match_features, norm=cv2.NORM_HAMMING, threshold=50),
    )


def match_features(
    first: List[Feature], second: List[Feature], norm, threshold
) -> List[Match]:
    matcher = cv2.BFMatcher_create(normType=norm, crossCheck=True)
    if not first or not second:
        return []
    first_descriptors = np.array(list(map(lambda x: x.descriptor, first)))
    second_descriptors = np.array(list(map(lambda x: x.descriptor, second)))
    matches = matcher.match(first_descriptors, second_descriptors)
    return [
        (match.queryIdx, match.trainIdx)
        for match in matches
        if match.distance <= threshold
    ]


def is_in_view(
    landmarks: Dict[int, Landmark],
    T_cam_obj: np.ndarray,
    params: CameraParameters,
    min_landmarks: int = 1,
):
    # constant if all landmarks in view
    num_in_view = 0
    for landmark in landmarks.values():
        pt_3d_cam = from_homogeneous(T_cam_obj @ to_homogeneous(landmark.pt_3d))
        x, y = project(params, pt_3d_cam)
        if x > 0 and y > 0 and x < 2 * params.cx and y < 2 * params.cy:
            num_in_view += 1
            if num_in_view == min_landmarks:
                # at least one landmark is in view
                return True
    return False


def get_masks_from_landmarks(
    landmarks: List[Landmark],
    T_cam_obj: np.ndarray,
    stereo_cam: StereoCamera,
    img_shape: Tuple[int, int],
    num_landmarks: int = 30,
):
    if len(landmarks) > num_landmarks:
        rng = np.random.default_rng()
        lm_subset = rng.choice(list(landmarks), size=num_landmarks, replace=False)
    else:
        lm_subset = landmarks
    left_points = []
    right_points = []
    for lmid in lm_subset:
        lm = landmarks[lmid]
        pt_3d_left_cam = from_homogeneous(T_cam_obj @ to_homogeneous(lm.pt_3d))
        pt_3d_right_cam = from_homogeneous(
            np.linalg.inv(stereo_cam.T_left_right)
            @ T_cam_obj
            @ to_homogeneous(lm.pt_3d)
        )
        x_left, y_left = project(stereo_cam.left, pt_3d_left_cam)
        x_right, y_right = project(stereo_cam.right, pt_3d_right_cam)
        if (
            x_left > 0
            and y_left > 0
            and x_left < img_shape[1]
            and y_left < img_shape[0]
        ):
            left_points.append([y_left, x_left])
        if (
            x_right > 0
            and y_right > 0
            and x_right < img_shape[1]
            and y_left < img_shape[0]
        ):
            right_points.append([y_right, x_right])
    if len(left_points) < 3:
        left_mask = None
    else:
        left_convex_hull = get_convex_hull_mask(left_points, img_shape)
        left_mask = fill_contours(left_convex_hull)
        num_pixels_left = min(10, max(1, 1000 // (1 + int(left_mask.sum()))))
        left_mask = dilate_mask(left_mask, num_pixels_left)
        if not np.any(left_mask):
            left_mask = None
    if len(right_points) < 3:
        right_mask = None
    else:
        right_convex_hull = get_convex_hull_mask(right_points, img_shape)
        right_mask = fill_contours(right_convex_hull)
        num_pixels_right = min(10, max(1, 1000 // (1 + int(right_mask.sum()))))
        right_mask = dilate_mask(right_mask, num_pixels_right)
        if not np.any(right_mask):
            right_mask = None
    return (left_mask, right_mask)


def fill_contours(arr):
    return (
        np.maximum.accumulate(arr, 1) & np.maximum.accumulate(arr[:, ::-1], 1)[:, ::-1]
    )


def triangulate(
    vec_left: np.ndarray,
    vec_right: np.ndarray,
    R_left_right: np.ndarray,
    t_left_right: np.ndarray,
) -> np.ndarray:
    # TODO: better triangulate function
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


def draw_features(img: np.ndarray, features: List[Feature]) -> np.ndarray:
    keypoints = [cv2.KeyPoint(x=f.u, y=f.v, _size=1) for f in features]
    return cv2.drawKeypoints(img, keypoints, None)


def triangulate_stereo_match(left_feature, right_feature, stereo_cam, T_ref_cam=None):
    left_pt = np.array([left_feature.u, left_feature.v])
    right_pt = np.array([right_feature.u, right_feature.v])
    if not np.allclose(left_feature.v, right_feature.v, atol=1):
        # match doesn't fullfill epipolar constraint
        raise TriangulationError("Epipolar constraint violated")
    vec_left = back_project(stereo_cam.left, left_pt)
    vec_right = back_project(stereo_cam.right, right_pt)
    R_left_right = stereo_cam.T_left_right[:3, :3]
    t_left_right = stereo_cam.T_left_right[:3, 3].reshape(3, 1)
    try:
        pt_3d_left_cam = triangulate(vec_left, vec_right, R_left_right, t_left_right)
    except np.linalg.LinAlgError:
        raise TriangulationError("Triangulation failed numerically")
    if pt_3d_left_cam[-1] < 0.5 or np.linalg.norm(pt_3d_left_cam) > config.MAX_DIST:
        # triangulated point should not be behind camera (or very close) or too far away
        raise TriangulationError("Too close or too far from camera")
    if T_ref_cam is not None:
        return from_homogeneous(T_ref_cam @ to_homogeneous(pt_3d_left_cam))
    else:
        return pt_3d_left_cam


def draw_contours(mask, img, color, thickness=1):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(img, contours, -1, color, thickness)
