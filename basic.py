import argparse
import glob
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from bamot.core.base_types import Camera
from bamot.core.preprocessing import preprocess_frame


def detect_objects(
    stereo_images: Tuple[np.ndarray, np.ndarray],
    img_idx: int,
    gt_data: Dict[int, List[Dict[str, float]]],
) -> List[Dict[str, float]]:
    return gt_data.get(img_idx, [])


def detect_features(img: np.ndarray, mask_percentage: float) -> List[Dict[str, float]]:
    orb = cv2.ORB_create(nfeatures=1000)
    detected_features = []
    if np.count_nonzero(img) == 0:
        return []
    y_center, x_center, _ = np.mean(img.nonzero(), axis=1)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    img_small = cv2.resize(img, (0, 0), fx=mask_percentage, fy=mask_percentage)
    x_min = mask.shape[1] - img_small.shape[1]
    x_max = x_min + img_small.shape[1]
    y_min = mask.shape[0] - img_small.shape[0]
    y_max = y_min + img_small.shape[0]
    mask[y_min:y_max, x_min:x_max] = img_small[:, :, 0]
    y_center_mask, x_center_mask = np.mean(mask.nonzero(), axis=1)
    y_shift = int(y_center - y_center_mask)
    x_shift = int(x_center - x_center_mask)
    mask_roll = np.roll(mask, y_shift, axis=0)
    mask_roll = np.roll(mask, x_shift, axis=1)
    mask_roll[mask_roll != 0] = 255
    # cv2.imshow("img", img)
    # cv2.waitKey()
    # cv2.imshow("img small", img_small)
    # cv2.imshow("mask", mask)
    # cv2.waitKey()
    # cv2.imshow("mask roll", mask_roll)
    # cv2.waitKey()
    # mask = np.zeros(img.shape[:2], dtype=np.uint8)
    # mask[50:-50, 50:-50] = 255
    # mask[150:-150, 150:-150] = 255
    # mask[300:-300, 300:-300] = 255
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # mask = None
    # obj_center = np.mean(mask.nonzero(), axis=

    features, descriptors = orb.detectAndCompute(img, mask=mask_roll)
    if features:
        for feature, descriptor in zip(features, descriptors):
            u, v = feature.pt
            detected_features.append({"u": u, "v": v, "descriptor": descriptor})

    return detected_features


def get_gt_bounding_boxes_from_kitti(
    label_file: Path,
) -> Dict[int, List[Dict[str, float]]]:
    objects = {}
    with open(label_file.as_posix(), "r") as fd:
        curr_frame = 0
        objects_in_frame: List[Dict[str, float]] = []
        for line in fd:
            cols = line.split(" ")
            frame = int(cols[0])
            typ = cols[2]
            if typ == "DontCare" or typ == "Misc":
                continue
            if frame != curr_frame:
                objects[curr_frame] = objects_in_frame.copy()
                objects_in_frame.clear()
                curr_frame = frame
            w, h, l = list(map(float, cols[10:13]))
            x, y, z = list(map(float, cols[13:16]))
            bbox = list(map(float, cols[6:10]))
            ry = float(cols[16])
            objects_in_frame.append(
                {"x": x, "y": y, "z": z, "width": w, "height": h, "length": l, "ry": ry}
            )
        objects[curr_frame] = objects_in_frame
    return objects


def project(pt_3d_cam: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    return intrinsics @ pt_3d_cam


def back_project(pt_2d: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    mx = (float(pt_2d[0]) - cx) / fx
    my = (float(pt_2d[1]) - cy) / fy
    length = np.sqrt(mx ** 2 + my ** 2 + 1)
    return (np.array([mx, my, 1]) / length).reshape(3, 1)


def get_cameras_from_kitti(calib_file: Path) -> Tuple[Camera, Camera]:
    with open(calib_file.as_posix(), "r") as fd:
        for line in fd:
            cols = line.split(" ")
            name = cols[0]
            if "R_02" in name:
                R02 = np.array(list(map(float, cols[1:]))).reshape(3, 3)
            elif "T_02" in name:
                t02 = np.array(list(map(float, cols[1:]))).reshape(3)
            elif "R_03" in name:
                R03 = np.array(list(map(float, cols[1:]))).reshape(3, 3)
            elif "T_03" in name:
                t03 = np.array(list(map(float, cols[1:]))).reshape(3)
            elif "P_rect_02" in name:
                intrinsics_02 = np.array(list(map(float, cols[1:]))).reshape(3, 4)
            elif "P_rect_03" in name:
                intrinsics_03 = np.array(list(map(float, cols[1:]))).reshape(3, 4)

    T02 = np.identity(4)
    T02[:3, :3] = R02
    T02[:3, 3] = t02
    T03 = np.identity(4)
    T03[:3, :3] = R03
    T03[:3, 3] = t03
    T23 = np.linalg.inv(T02) @ T03
    left_cam = Camera(
        T_0_i=np.identity(4),
        project=partial(project, intrinsics=intrinsics_02),
        back_project=partial(back_project, intrinsics=intrinsics_02),
    )
    right_cam = Camera(
        T_0_i=T23,
        project=partial(project, intrinsics=intrinsics_02),
        back_project=partial(back_project, intrinsics=intrinsics_03),
    )
    return left_cam, right_cam


if __name__ == "__main__":
    # test on single kitti image
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-convex-hull", action="store_true")
    parser.add_argument("--object-mask", type=float, default=0.9)
    args = parser.parse_args()
    base_path = Path(__file__).parent / "data" / "KITTI"
    label_file = base_path / "tracking" / "training" / "label_02" / "0001.txt"
    calib_file = base_path / "calib_cam_to_cam.txt"
    left_img_path = base_path / "tracking" / "training" / "image_02" / "0001"
    right_img_path = base_path / "tracking" / "training" / "image_03" / "0001"
    left_imgs = sorted(glob.glob(left_img_path.as_posix() + "/*.png"))
    right_imgs = sorted(glob.glob(right_img_path.as_posix() + "/*.png"))
    bounding_boxes = get_gt_bounding_boxes_from_kitti(label_file)
    for idx, (l, r) in enumerate(zip(left_imgs, right_imgs)):
        left_img = cv2.imread(l, cv2.IMREAD_COLOR)
        right_img = cv2.imread(r, cv2.IMREAD_COLOR)
        left_cam, right_cam = get_cameras_from_kitti(calib_file)
        detect_objects_fn = partial(detect_objects, img_idx=idx, gt_data=bounding_boxes)

        detect_features_fn = partial(detect_features, mask_percentage=args.object_mask)
        objects, (masked_left_image, masked_right_image) = preprocess_frame(
            (left_img, right_img),
            (left_cam, right_cam),
            detect_objects_fn=detect_objects_fn,
            detect_features_fn=detect_features_fn,
            use_feature_convex_hull=args.feature_convex_hull,
        )
        # get keypoints
        left_keypoints = []
        right_keypoints = []
        for obj in objects:
            for feature in obj.left_features:
                left_keypoints.append(cv2.KeyPoint(x=feature.u, y=feature.v, _size=1))
            for feature in obj.right_features:
                right_keypoints.append(cv2.KeyPoint(x=feature.u, y=feature.v, _size=1))
            for point in obj.left_projected_bounding_box:
                for other_point in obj.left_projected_bounding_box:
                    masked_left_image = cv2.line(
                        masked_left_image,
                        tuple(map(int, point)),
                        tuple(map(int, other_point)),
                        color=(255, 0, 0),
                    )

        masked_left_image = cv2.drawKeypoints(masked_left_image, left_keypoints, None)
        masked_right_image = cv2.drawKeypoints(
            masked_right_image, right_keypoints, None
        )
        total = np.vstack([masked_left_image, masked_right_image])
        cv2.imshow("result", total)
        cv2.waitKey()
