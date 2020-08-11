"""Script for BAMOT with GT data from KITTI and mocked SLAM system.
"""
import argparse
import glob
import logging
import queue
import time
from pathlib import Path
from threading import Event, Thread
from typing import Iterable, List

import cv2
import numpy as np

from bamot.core.base_types import StereoImage, StereoObjectDetection
from bamot.core.mot import run
from bamot.util.cv import get_orb_feature_matcher
from bamot.util.kitti import get_cameras_from_kitti
from bamot.util.viewer import run as run_viewer

LOG_LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "ERROR": logging.ERROR}
LOGGER = logging.getLogger("MAIN")


def _fake_slam(slam_data: queue.Queue, stop_flag: Event):
    all_poses = []
    while not stop_flag.is_set():
        all_poses.append(np.identity(4))
        slam_data.put(all_poses)  # not moving
        time.sleep(20 / 1000)  # 20 ms or 50 Hz
    LOGGER.debug("Finished yielding fake slam data")


def _get_image_stream(
    kitti_path: Path, scene: str, stop_flag: Event
) -> Iterable[StereoImage]:
    left_img_path = kitti_path / "image_02" / scene
    right_img_path = kitti_path / "image_03" / scene
    left_imgs = sorted(glob.glob(left_img_path.as_posix() + "/*.png"))
    right_imgs = sorted(glob.glob(right_img_path.as_posix() + "/*.png"))
    LOGGER.debug(
        "Found %d left images and %d right images", len(left_imgs), len(right_imgs)
    )
    for left, right in zip(left_imgs, right_imgs):
        left_img = cv2.imread(left, cv2.IMREAD_COLOR).astype(np.uint8)
        right_img = cv2.imread(right, cv2.IMREAD_COLOR).astype(np.uint8)
        yield StereoImage(left_img, right_img)
    LOGGER.debug("Setting stop flag")
    stop_flag.set()


def _get_detection_stream(
    obj_detections_path: Path,
) -> Iterable[List[StereoObjectDetection]]:
    detection_files = sorted(glob.glob(obj_detections_path.as_posix() + "/*.json"))
    LOGGER.debug("Found %d detection files", len(detection_files))
    for f in detection_files:
        with open(f, "r") as fd:
            json_data = fd.read()
        detections = StereoObjectDetection.schema().loads(json_data, many=True)
        yield detections
    LOGGER.debug("Finished yielding object detections")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("BAMOT with GT KITTI data")
    parser.add_argument(
        "-v",
        "--verbosity",
        dest="verbosity",
        help="verbosity of output (default is INFO)",
        type=str,
        choices=["DEBUG", "INFO", "ERROR"],
        default="INFO",
    )
    parser.add_argument(
        "-i",
        "--images",
        dest="images",
        help="path to kitti tracking dataset (default is `./data/KITTI/tracking/training`",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--detections",
        dest="detections",
        help="path to detections generated with `preprocess_kitti_groundtruth.py`"
        + "(default is `./data/KITTI/tracking/training/preprocessed/mot",
        type=str,
    )
    parser.add_argument(
        "-s",
        "--scene",
        dest="scene",
        help="scene to run (default is 1)",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    scene = str(args.scene).zfill(4)
    if args.images is not None:
        kitti_path = Path(args.images)
    else:
        kitti_path = Path(".") / "data" / "KITTI" / "tracking" / "training"
    if args.detections:
        obj_detections_path = Path(args.detections)
    else:
        obj_detections_path = kitti_path / "preprocessed" / "mot" / scene
    print("STARTING BAMOT with GT KITTI data")
    print(f"KITTI PATH: {kitti_path}")
    print(f"SCENE: {scene}")
    print(f"OBJ. DET. PATH: {obj_detections_path}")
    print(f"VERBOSITY LEVEL: {args.verbosity}")
    logging.basicConfig(level=LOG_LEVELS[args.verbosity])
    shared_data: queue.Queue = queue.Queue()
    slam_data: queue.Queue = queue.Queue()
    feature_matcher = get_orb_feature_matcher()
    stop_flag = Event()
    next_step = Event()
    next_step.set()
    image_stream = _get_image_stream(kitti_path, scene, stop_flag)
    stereo_cam = get_cameras_from_kitti(kitti_path / "calib_cam_to_cam.txt")
    detection_stream = _get_detection_stream(obj_detections_path)
    slam_thread = Thread(
        target=_fake_slam, args=[slam_data, stop_flag], name="Fake SLAM Thread"
    )
    mot_thread = Thread(
        target=run,
        kwargs={
            "images": image_stream,
            "detections": detection_stream,
            "feature_matcher": feature_matcher,
            "stereo_cam": stereo_cam,
            "slam_data": slam_data,
            "shared_data": shared_data,
            "stop_flag": stop_flag,
            "next_step": next_step,
        },
        name="MOT Thread",
    )
    LOGGER.debug("Starting fake SLAM thread")
    slam_thread.start()
    LOGGER.debug("Starting MOT thread")
    mot_thread.start()
    LOGGER.debug("Starting viewer")
    run_viewer(shared_data=shared_data, stop_flag=stop_flag, next_step=next_step)
    slam_thread.join()
    LOGGER.debug("Joined fake SLAM thread")
    mot_thread.join()
    LOGGER.debug("Joined MOT thread")
    shared_data.join()
    print("FINISHED RUNNING KITTI GT MOT")
