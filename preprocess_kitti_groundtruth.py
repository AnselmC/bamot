"""Preprocesses kitti GT data.
"""
import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import tqdm

from bamot.config import CONFIG as config
from bamot.core.base_types import StereoImage
from bamot.core.preprocessing import preprocess_frame
from bamot.util.kitti import (get_cameras_from_kitti,
                              get_estimated_obj_detections,
                              get_gt_obj_detections_from_kitti,
                              get_image_stream)
from bamot.util.viewer import get_screen_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        help="the scene to preprocess, default is 0",
        choices=list(map(str, range(0, 21))) + ["all"],
        nargs="*",
        default=[0],
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
        "--no-right",
        help="disable using precomputed right obj masks (transfer left masks to right)",
        action="store_true",
    )
    parser.add_argument(
        "--no-view",
        help="flag to disable viewing the preprocessed data while it's being generated (quit execution by hitting `q`)",
        action="store_true",
    )
    parser.add_argument(
        "-n",
        "--no-continue",
        action="store_true",
        help="Disable continuous stream (press `n` for next frame)",
    )

    args = parser.parse_args()
    scenes = args.s if "all" not in args.s else range(0, 20)

    for scene in tqdm.tqdm(scenes, position=0):
        scene = str(scene).zfill(4)
        kitti_path = Path(config.KITTI_PATH)
        if not args.no_save:
            if not args.o:
                save_path = kitti_path / "preprocessed"
            else:
                save_path = Path(args.o)
            save_path_slam = save_path / "slam"
            save_path_slam_left = save_path_slam / "image_02" / scene
            save_path_slam_right = save_path_slam / "image_03" / scene
            save_path_mot = save_path / "mot" / scene
            save_path_slam_left.mkdir(parents=True, exist_ok=True)
            save_path_slam_right.mkdir(parents=True, exist_ok=True)
            save_path_mot.mkdir(parents=True, exist_ok=True)

        stereo_cam, T02 = get_cameras_from_kitti(kitti_path)
        stereo_cam.T_left_right[0, 3] = 0.03
        image_stream = get_image_stream(kitti_path, scene, with_file_names=True)
        if not args.no_view:
            width, height = get_screen_size()
            cv2.namedWindow("Preprocessed", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Preprocessed", (width // 2, height // 2))

        if not args.no_right:
            all_right_object_detections = get_estimated_obj_detections(
                kitti_path, scene
            )
        else:
            all_right_object_detections = {}
        for idx, (stereo_image, filenames) in tqdm.tqdm(
            enumerate(image_stream), total=len(image_stream), position=1,
        ):
            right_object_detections = all_right_object_detections.get(idx)
            left_fname, right_fname = filenames
            left_object_detections = get_gt_obj_detections_from_kitti(
                kitti_path, scene, idx
            )
            masked_stereo_image_slam, stereo_object_detections = preprocess_frame(
                stereo_image,
                stereo_cam,
                left_object_detections=left_object_detections,
                right_object_detections=right_object_detections,
            )
            left_mot_mask = masked_stereo_image_slam.left == 0
            right_mot_mask = masked_stereo_image_slam.right == 0
            left_img_mot = np.zeros(stereo_image.left.shape, dtype=np.uint8)
            left_img_mot[left_mot_mask] = stereo_image.left[left_mot_mask]
            right_img_mot = np.zeros(stereo_image.right.shape, dtype=np.uint8)
            right_img_mot[right_mot_mask] = stereo_image.right[right_mot_mask]
            masked_stereo_image_mot = StereoImage(left_img_mot, right_img_mot)
            result_slam = np.hstack(
                [masked_stereo_image_slam.left, masked_stereo_image_slam.right]
            )
            result_slam = cv2.putText(
                result_slam,
                left_fname,
                (5, 25),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 0),
                thickness=2,
            )
            result_slam = cv2.putText(
                result_slam,
                right_fname,
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
                pressed_key = cv2.waitKey(1)
                if args.no_continue:
                    while pressed_key not in [ord("q"), ord("n")]:
                        pressed_key = cv2.waitKey(1)
                if pressed_key == ord("q"):
                    cv2.destroyAllWindows()
                    break
            if not args.no_save:
                img_name = left_fname.split("/")[-1]
                img_id = img_name.split(".")[0]
                slam_left_path = save_path_slam_left / img_name
                slam_right_path = save_path_slam_right / img_name
                cv2.imwrite(slam_left_path.as_posix(), masked_stereo_image_slam.left)
                cv2.imwrite(slam_right_path.as_posix(), masked_stereo_image_slam.right)
                obj_det_path = (save_path_mot / img_id).as_posix() + ".pkl"
                with open(obj_det_path, "wb") as fp:
                    pickle.dump(stereo_object_detections, fp)
