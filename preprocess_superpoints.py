import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import tqdm

from bamot.config import CONFIG as config
from bamot.util.cv import draw_features, get_superpoint_feature_matcher
from bamot.util.kitti import (get_cameras_from_kitti, get_detection_stream,
                              get_image_stream)
from bamot.util.viewer import get_screen_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        help="the scene to preprocess, default is 0",
        choices=list(map(str, range(0, 20))) + ["all"],
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
        "--no-view",
        help="flag to disable viewing the preprocessed data while it's being generated (quit execution by hitting `q`)",
        action="store_true",
    )
    parser.add_argument(
        "--no-continuous",
        "--nc",
        dest="nc",
        help="Don't process continuously, press `n` for next step",
        action="store_true",
    )

    args = parser.parse_args()
    scenes = args.s if "all" not in args.s else range(0, 20)
    continuous = not args.nc
    for scene in tqdm.tqdm(scenes, position=0):
        scene = str(scene).zfill(4)
        kitti_path = Path(config.KITTI_PATH)
        if not args.no_save:
            if not args.o:
                save_path = kitti_path / "preprocessed"
            else:
                save_path = Path(args.o)
            save_path = save_path / "superpoint" / scene
            save_path.mkdir(parents=True, exist_ok=True)

        stereo_cam, T02 = get_cameras_from_kitti(kitti_path)
        image_stream = get_image_stream(kitti_path, scene)
        obj_detections_path = Path(config.DETECTIONS_PATH) / scene
        detection_stream = get_detection_stream(obj_detections_path, 0)
        if not args.no_view:
            width, height = get_screen_size()
            cv2.namedWindow("Preprocessed", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Preprocessed", (width // 2, height // 2))
        feature_matcher = get_superpoint_feature_matcher()
        for idx, (stereo_image, detections) in tqdm.tqdm(
            enumerate(zip(image_stream, detection_stream)),
            total=len(image_stream),
            position=1,
        ):
            left_output = stereo_image.left
            right_output = stereo_image.right
            curr_img_path = save_path / str(idx).zfill(6)
            for detection in detections:
                left_features = feature_matcher.detect_features(
                    stereo_image.left, detection.left.mask
                )
                left_output = draw_features(left_output, left_features)
                right_features = feature_matcher.detect_features(
                    stereo_image.right, detection.right.mask
                )
                right_output = draw_features(right_output, right_features)
                track_id = detection.left.track_id
                curr_track_path = curr_img_path / str(track_id).zfill(4)
                curr_track_path.mkdir(exist_ok=True, parents=True)
                if not args.no_save:
                    left_path = curr_track_path / "left.pkl"
                    right_path = curr_track_path / "right.pkl"
                    with open(left_path.as_posix(), "wb") as fp:
                        pickle.dump(left_features, fp)
                    with open(right_path.as_posix(), "wb") as fp:
                        pickle.dump(right_features, fp)
            if not args.no_view:
                full_img = np.hstack([left_output, right_output])
                cv2.imshow("Preprocessed", full_img)
                keypress = cv2.waitKey(1)
                if keypress == ord("q"):
                    cv2.destroyAllWindows()
                    break
                if not continuous:
                    while keypress != ord("n"):
                        keypress = cv2.waitKey(1)
