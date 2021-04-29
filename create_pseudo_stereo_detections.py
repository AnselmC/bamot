"""Preprocesses kitti GT data.
"""
import argparse
import multiprocessing as mp
import os
from collections import defaultdict
from functools import wraps
from pathlib import Path

import cv2
import numpy as np
import tqdm

from bamot.config import CONFIG as config
from bamot.core.base_types import StereoImage
from bamot.core.preprocessing import preprocess_frame
from bamot.util.kitti import (get_2d_track_line, get_cameras_from_kitti,
                              get_estimated_obj_detections,
                              get_gt_obj_detections_from_kitti,
                              get_image_stream)
from bamot.util.misc import get_color
from bamot.util.viewer import get_screen_size


def except_all(func):
    @wraps(func)
    def wrapper(scene):
        try:
            return func(scene)
        except Exception as e:
            print("scene: ", scene)
            print(e)

    return wrapper


@except_all
def _process_scene(scene):
    scene = str(scene).zfill(4)
    kitti_path = Path(config.KITTI_PATH)
    if not args.no_save:
        if args.o:
            save_path = Path(args.o)
        else:
            save_path = kitti_path
        if args.use_gt:
            save_path_detections = save_path / "stereo_detections_gt"
        else:
            save_path_detections = save_path / "stereo_detections"
        if args.save_slam:
            if args.use_gt:
                save_path_slam = save_path / "masked_slam_input_gt"
            else:
                save_path_slam = save_path / "masked_slam_input"
            save_path_slam_left = save_path_slam / "image_02" / scene
            save_path_slam_right = save_path_slam / "image_03" / scene
            save_path_slam_left.mkdir(parents=True, exist_ok=True)
            save_path_slam_right.mkdir(parents=True, exist_ok=True)
        save_path_mot_left = save_path_detections / "image_02"
        save_path_mot_right = save_path_detections / "image_03"
        left_mot_out_file = save_path_mot_left / (scene + ".txt")
        if left_mot_out_file.exists():
            left_mot_out_file.unlink()  # overwrite file
        right_mot_out_file = save_path_mot_right / (scene + ".txt")
        if right_mot_out_file.exists():
            right_mot_out_file.unlink()  # overwrite file
        save_path_mot_left.mkdir(parents=True, exist_ok=True)
        save_path_mot_right.mkdir(parents=True, exist_ok=True)

    stereo_cam, T02 = get_cameras_from_kitti(kitti_path, scene)
    stereo_cam.T_left_right[0, 3] = 0.03
    image_stream = get_image_stream(kitti_path, scene, with_file_names=True)
    if not args.no_view:
        width, height = get_screen_size()
        cv2.namedWindow("Preprocessed", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Preprocessed", (width // 2, height // 2))

    if not args.no_right:
        all_right_object_detections = get_estimated_obj_detections(
            kitti_path, scene, "right"
        )
    else:
        all_right_object_detections = {}
    if not args.use_gt:
        all_left_object_detections = get_estimated_obj_detections(
            kitti_path, scene, "left"
        )

    colors = defaultdict(lambda: get_color(normalized=False, as_tuple=True))
    for idx, (stereo_image, filenames) in tqdm.tqdm(
        enumerate(image_stream),
        total=len(image_stream),
        position=int(scene),
        leave=False,
    ):
        right_object_detections = all_right_object_detections.get(idx, [])
        left_fname, right_fname = filenames
        if args.use_gt:
            left_object_detections = get_gt_obj_detections_from_kitti(
                kitti_path, scene, idx
            )
        else:
            left_object_detections = all_left_object_detections.get(idx, [])
        masked_stereo_image_slam, stereo_object_detections = preprocess_frame(
            stereo_image,
            left_object_detections=left_object_detections,
            right_object_detections=right_object_detections,
            only_iou=args.only_iou,
            use_unmatched=args.use_unmatched,
            colors=colors,
        )
        left_mot_mask = masked_stereo_image_slam.left == 0
        right_mot_mask = masked_stereo_image_slam.right == 0
        left_img_mot = 255 * np.ones(stereo_image.left.shape, dtype=np.uint8)
        left_img_mot[left_mot_mask] = stereo_image.left[left_mot_mask]
        right_img_mot = 255 * np.ones(stereo_image.right.shape, dtype=np.uint8)
        right_img_mot[right_mot_mask] = stereo_image.right[right_mot_mask]
        masked_stereo_image_mot = StereoImage(
            left_img_mot,
            right_img_mot,
            img_width=stereo_image.img_width,
            img_height=stereo_image.img_height,
        )
        result_slam = np.hstack(
            [masked_stereo_image_slam.left, masked_stereo_image_slam.right]
        )
        result_slam = cv2.putText(
            result_slam,
            "/".join(left_fname.split("/")[-3:]),
            (5, 25),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 255, 0),
            thickness=2,
        )
        result_slam = cv2.putText(
            result_slam,
            "/".join(right_fname.split("/")[-3:]),
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
            if args.save_slam:
                slam_left_path = save_path_slam_left / img_name
                slam_right_path = save_path_slam_right / img_name
                cv2.imwrite(slam_left_path.as_posix(), masked_stereo_image_slam.left)
                cv2.imwrite(slam_right_path.as_posix(), masked_stereo_image_slam.right)
            for detection in stereo_object_detections:
                track_id = detection.left.track_id
                obj_cls = detection.left.cls
                img_height = stereo_image.img_height
                img_width = stereo_image.img_width

                left_track_line = get_2d_track_line(
                    img_id=img_id,
                    track_id=track_id,
                    obj_cls=obj_cls,
                    height=img_height,
                    width=img_width,
                    mask=detection.left.mask,
                )
                right_track_line = get_2d_track_line(
                    img_id=img_id,
                    track_id=track_id,
                    obj_cls=obj_cls,
                    height=img_height,
                    width=img_width,
                    mask=detection.right.mask,
                )

                with open(left_mot_out_file, "a") as fp:
                    fp.write(left_track_line + "\n")

                with open(right_mot_out_file, "a") as fp:
                    fp.write(right_track_line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        help="the scene to preprocess, default is 0",
        choices=list(map(str, range(0, 28))) + ["all"],
        nargs="*",
        default=[0],
    )
    parser.add_argument(
        "-o",
        help="where to save the masked data (default is `{KITTI_TRACKING_PATH}/{KITTI_TRACKING_SUBSET}/preprocessed_{est | gt}` depending on whether `--use-gt` is set)",
        type=str,
    )
    parser.add_argument(
        "--no-save",
        help="flag to disable saving of preprocessed data",
        action="store_true",
    )
    parser.add_argument(
        "--save-slam",
        help="flag to store masked images for static SLAM (disabled with `--no-save`) ",
        action="store_true",
    )
    parser.add_argument(
        "--no-right",
        help="disable using estimated right obj masks (transfer left masks to right)",
        action="store_true",
    )
    parser.add_argument(
        "--use-gt",
        help="Use ground truth object masks for left image",
        action="store_true",
    )
    parser.add_argument(
        "--use-unmatched",
        help="Use unmatched tracks by translating mask to other image and dilating it.",
        action="store_true",
    )
    parser.add_argument(
        "--only-iou",
        help="Use only Intersection-over-Union as matching heuristic between left and right detections",
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
    parser.add_argument(
        "--use-right-tracks",
        action="store_true",
        help="Use right track ids (default uses left)",
    )
    parser.add_argument(
        "--disable-mp",
        dest="disable_mp",
        help="Disable multiprocessing of scenes (automatically disabled w/o `--no-view`)",
    )

    args = parser.parse_args()
    scenes = args.s if "all" not in args.s else range(0, 28)
    num_processes = os.cpu_count() if (not args.disable_mp) and args.no_view else 1

    if num_processes > 1:
        with mp.Manager() as manager:
            with manager.Pool(processes=num_processes,) as p:
                # TODO: overall pbar doesn't really work
                for scene in tqdm.tqdm(
                    p.imap(_process_scene, scenes),
                    total=len(scenes),
                    position=len(scenes) + 1,
                    nrows=len(scenes) + 1,
                ):
                    pass
    else:
        for scene in tqdm.tqdm(scenes, total=len(scenes), position=0):
            _process_scene(scene)
