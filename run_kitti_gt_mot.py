"""Script for BAMOT with GT data from KITTI and mocked SLAM system.
"""
import argparse
import glob
import logging
import multiprocessing as mp
import queue
import threading
import time
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import cv2
import numpy as np

from bamot.core.base_types import (ObjectDetection, StereoImage,
                                   StereoObjectDetection)
from bamot.core.mot import run
from bamot.util.cv import (get_orb_feature_matcher,
                           get_superpoint_feature_matcher)
from bamot.util.kitti import get_cameras_from_kitti
from bamot.util.viewer import run as run_viewer

LOG_LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "ERROR": logging.ERROR}
LOGGER = logging.getLogger("MAIN")


def _fake_slam(
    slam_data: Union[queue.Queue, mp.Queue], stop_flag: Union[threading.Event, mp.Event]
):
    all_poses = []
    while not stop_flag.is_set():
        all_poses.append(np.identity(4))
        slam_data.put(all_poses)  # not moving
        time.sleep(20 / 1000)  # 20 ms or 50 Hz
    LOGGER.debug("Finished yielding fake slam data")


def _get_image_shape(kitti_path: Path) -> Tuple[int, int]:
    left_img_path = kitti_path / "image_02" / scene
    left_imgs = sorted(glob.glob(left_img_path.as_posix() + "/*.png"))
    img_shape = cv2.imread(left_imgs[0], cv2.IMREAD_COLOR).astype(np.uint8).shape
    return img_shape


def _get_image_stream(
    kitti_path: Path,
    scene: str,
    stop_flag: Union[threading.Event, mp.Event],
    offset: int,
) -> Tuple[Iterable[StereoImage], Tuple[int, int]]:
    left_img_path = kitti_path / "image_02" / scene
    right_img_path = kitti_path / "image_03" / scene
    left_imgs = sorted(glob.glob(left_img_path.as_posix() + "/*.png"))
    right_imgs = sorted(glob.glob(right_img_path.as_posix() + "/*.png"))
    LOGGER.debug(
        "Found %d left images and %d right images", len(left_imgs), len(right_imgs)
    )
    LOGGER.debug("Starting at image %d", offset)
    for left, right in zip(left_imgs[offset:], right_imgs[offset:]):
        left_img = cv2.imread(left, cv2.IMREAD_COLOR).astype(np.uint8)
        right_img = cv2.imread(right, cv2.IMREAD_COLOR).astype(np.uint8)
        yield StereoImage(left_img, right_img)
    LOGGER.debug("Setting stop flag")
    stop_flag.set()


def _get_detection_stream(
    obj_detections_path: Path,
    offset: int,
    img_shape: Tuple[int, int],
    include_static_map: bool = False,
) -> Iterable[List[StereoObjectDetection]]:
    detection_files = sorted(glob.glob(obj_detections_path.as_posix() + "/*.json"))
    static_det = ObjectDetection(
        convex_hull=[
            (0, 0),
            (img_shape[1], 0),
            (0, img_shape[0]),
            (img_shape[1], img_shape[0]),
        ],
        track_id=-1,
    )
    LOGGER.debug("Found %d detection files", len(detection_files))
    for f in detection_files[offset:]:
        with open(f, "r") as fd:
            json_data = fd.read()
        detections = StereoObjectDetection.schema().loads(json_data, many=True)
        if include_static_map:
            detections.append(StereoObjectDetection(left=static_det, right=static_det))
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
    parser.add_argument(
        "-o",
        "--offset",
        dest="offset",
        help="The image number offset to use (i.e. start at image # offset instead of 0)",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-m", "--map", dest="map", help="Include static map", action="store_true"
    )
    parser.add_argument(
        "-nv",
        "--no-viewer",
        dest="no_viewer",
        help="disable viewer",
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--features",
        help="Which feature type to use (default: orb)",
        dest="features",
        choices=["orb", "superpoint"],
        type=str.lower,
        default="orb",
    )
    parser.add_argument(
        "-mp",
        "--multiprocessing",
        help="Whether to use multiple processes instead of multiple threads",
        dest="multiprocessing",
        action="store_true",
    )
    parser.add_argument(
        "-c",
        "--continuous",
        action="store_true",
        dest="continuous",
        help="Whether to run process continuously (default is next step via 'n' keypress",
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
    if args.features == "orb":
        feature_matcher = get_orb_feature_matcher()
    else:
        feature_matcher = get_superpoint_feature_matcher()
    print(30 * "+")
    print("STARTING BAMOT with GT KITTI data")
    print(f"KITTI PATH: {kitti_path}")
    print(f"SCENE: {scene}")
    print(f"OBJ. DET. PATH: {obj_detections_path}")
    print(f"FEATURE MATCHER: {feature_matcher.name}")
    print(f"VERBOSITY LEVEL: {args.verbosity}")
    print(f"USING MULTI-PROCESS: {args.multiprocessing}")
    print(30 * "+")
    logging.basicConfig(level=LOG_LEVELS[args.verbosity])
    if args.multiprocessing:

        queue_class = mp.JoinableQueue
        flag_class = mp.Event
        process_class = mp.Process
    else:

        queue_class = queue.Queue
        flag_class = threading.Event
        process_class = threading.Thread
    shared_data = queue_class()
    slam_data = queue_class()
    stop_flag = flag_class()
    next_step = flag_class()
    next_step.set()
    img_shape = _get_image_shape(kitti_path)
    image_stream = _get_image_stream(kitti_path, scene, stop_flag, offset=args.offset)
    stereo_cam = get_cameras_from_kitti(kitti_path / "calib_cam_to_cam.txt")
    detection_stream = _get_detection_stream(
        obj_detections_path,
        img_shape=img_shape,
        offset=args.offset,
        include_static_map=args.map,
    )

    slam_process = process_class(
        target=_fake_slam, args=[slam_data, stop_flag], name="Fake SLAM"
    )
    mot_process = process_class(
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
            "continuous": args.continuous,
        },
        name="BAMOT",
    )
    LOGGER.debug("Starting fake SLAM thread")
    slam_process.start()
    LOGGER.debug("Starting MOT thread")
    mot_process.start()
    LOGGER.debug("Starting viewer")
    if args.no_viewer:
        while not stop_flag.is_set():
            shared_data.get(block=True)
            next_step.set()
    else:
        run_viewer(shared_data=shared_data, stop_flag=stop_flag, next_step=next_step)
    slam_process.join()
    LOGGER.debug("Joined fake SLAM thread")
    mot_process.join()
    LOGGER.debug("Joined MOT thread")
    shared_data.join()
    print("FINISHED RUNNING KITTI GT MOT")
