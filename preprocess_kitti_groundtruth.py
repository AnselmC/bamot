"""Preprocesses kitti GT data.
"""
import argparse
import glob
import warnings
from pathlib import Path
from typing import List

warnings.filterwarnings(action="ignore")

import cv2
import numpy as np
import tqdm

from bamot.core.base_types import StereoImage, StereoObjectDetection
from bamot.core.preprocessing import preprocess_frame
from bamot.util.kitti import (get_cameras_from_kitti,
                              get_gt_obj_segmentations_from_kitti)
from bamot.util.viewer import get_screen_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        help="the scene to preprocess, default is 0",
        choices=list(map(str, range(0, 20))) + ["all"],
        nargs="*",
        default=0,
    )
    parser.add_argument(
        "-d",
        help="the path to the training data of kitti's tracking dataset"
        + " (default is `./data/KITTI/tracking/training`",
        type=str,
    )
    parser.add_argument(
        "-o",
        help="where to save the masked data (default is `./data/KITTI/tracking/training/preprocessed`)",
        type=str,
    )
    parser.add_argument(
        "--no-save",
        help="flag to disable saving of preprocessed data",
        action="store_true",
    )
    parser.add_argument(
        "--no-view",
        help="flag to disable viewing the preprocessed data while it's being generated (quit execution by hitting `q`)",
        action="store_true",
    )

    args = parser.parse_args()
    scenes = args.s if "all" not in args.s else range(0, 20)

    for scene in tqdm.tqdm(scenes, position=0):
        scene = str(scene).zfill(4)
        if not args.d:
            base_path = (
                Path(__file__).parent / "data" / "KITTI" / "tracking" / "training"
            )
        else:
            base_path = Path(args.d)
        if not args.no_save:
            if not args.o:
                save_path = base_path / "preprocessed"
            else:
                save_path = Path(args.o)
            save_path_slam = save_path / "slam"
            save_path_slam_left = save_path_slam / "image_02" / scene
            save_path_slam_right = save_path_slam / "image_03" / scene
            save_path_mot = save_path / "mot" / scene
            save_path_slam_left.mkdir(parents=True, exist_ok=True)
            save_path_slam_right.mkdir(parents=True, exist_ok=True)
            save_path_mot.mkdir(parents=True, exist_ok=True)

        instance_path = base_path / "instances" / scene
        calib_file = base_path / "calib_cam_to_cam.txt"
        left_img_path = base_path / "image_02" / scene
        right_img_path = base_path / "image_03" / scene
        left_imgs: List[str] = sorted(glob.glob(left_img_path.as_posix() + "/*.png"))
        right_imgs: List[str] = sorted(glob.glob(right_img_path.as_posix() + "/*.png"))
        instances: List[str] = sorted(glob.glob(instance_path.as_posix() + "/*.png"))
        stereo_cam, T02 = get_cameras_from_kitti(calib_file)
        if not args.no_view:
            width, height = get_screen_size()
            cv2.namedWindow("Preprocessed", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Preprocessed", (width // 2, height // 2))
        for idx, (l, r, instance_file) in tqdm.tqdm(
            enumerate(zip(left_imgs, right_imgs, instances)),
            total=len(instances),
            position=1,
        ):
            left_img = cv2.imread(l, cv2.IMREAD_COLOR)
            right_img = cv2.imread(r, cv2.IMREAD_COLOR)
            object_detections = get_gt_obj_segmentations_from_kitti(instance_file)
            masked_stereo_image_slam, stereo_object_detections = preprocess_frame(
                StereoImage(left_img, right_img),
                stereo_cam,
                object_detections=object_detections,
            )
            left_mot_mask = masked_stereo_image_slam.left == 0
            right_mot_mask = masked_stereo_image_slam.right == 0
            left_img_mot = np.zeros(left_img.shape, dtype=np.uint8)
            left_img_mot[left_mot_mask] = left_img[left_mot_mask]
            right_img_mot = np.zeros(right_img.shape, dtype=np.uint8)
            right_img_mot[right_mot_mask] = right_img[right_mot_mask]
            masked_stereo_image_mot = StereoImage(left_img_mot, right_img_mot)
            result_slam = np.hstack(
                [masked_stereo_image_slam.left, masked_stereo_image_slam.right]
            )
            result_slam = cv2.putText(
                result_slam,
                l,
                (5, 25),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 0),
                thickness=2,
            )
            result_slam = cv2.putText(
                result_slam,
                r,
                (5 + right_img_mot.shape[1], 25),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 0),
                thickness=2,
            )
            result_mot = np.hstack(
                [masked_stereo_image_mot.left, masked_stereo_image_mot.right]
            )
            result = np.vstack([result_slam, result_mot])
            if not args.no_view:
                cv2.imshow("Preprocessed", result)
                if cv2.waitKey(1) == ord("q"):
                    cv2.destroyAllWindows()
                    break
            if not args.no_save:
                img_name = l.split("/")[-1]
                img_id = img_name.split(".")[0]
                slam_left_path = save_path_slam_left / img_name
                slam_right_path = save_path_slam_right / img_name
                cv2.imwrite(slam_left_path.as_posix(), masked_stereo_image_slam.left)
                cv2.imwrite(slam_right_path.as_posix(), masked_stereo_image_slam.right)
                obj_det_json = StereoObjectDetection.schema().dumps(
                    stereo_object_detections, many=True, indent=4
                )
                obj_det_path = (save_path_mot / img_id).as_posix() + ".json"
                with open(obj_det_path, "w") as fd:
                    fd.write(obj_det_json)
