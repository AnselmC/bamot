import argparse
import datetime
import json
import logging
import multiprocessing as mp
import queue
import subprocess
import threading
from pathlib import Path

import colorlog
from bamot.config import CONFIG as config
from bamot.config import get_config_dict
from bamot.core.disparity import run
from bamot.util.kitti import (get_cameras_from_kitti, get_detection_stream,
                              get_gt_poses_from_kitti, get_image_shape,
                              get_label_data_from_kitti)
from bamot.util.misc import TqdmLoggingHandler
from bamot.util.viewer import run as run_viewer
from run_kitti_gt_mot import _get_image_stream

LOG_LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "ERROR": logging.ERROR}
LOGGER = colorlog.getLogger()
HANDLER = TqdmLoggingHandler()
HANDLER.setFormatter(
    colorlog.ColoredFormatter(
        "%(asctime)s | %(log_color)s%(name)s:%(levelname)s%(reset)s: %(message)s",
        datefmt="%H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "white",
            "SUCCESS:": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
)
LOGGER.handlers = [HANDLER]

if __name__ == "__main__":
    parser = argparse.ArgumentParser("DISPARITY with GT KITTI data")
    parser.add_argument(
        "-v",
        "--verbosity",
        dest="verbosity",
        help="verbosity of output (default is INFO, surpresses most logs)",
        type=str,
        choices=["DEBUG", "INFO", "ERROR"],
        default="INFO",
    )
    parser.add_argument(
        "-s",
        "--scene",
        dest="scene",
        help="scene to run (default is 0)",
        choices=range(0, 21),
        type=int,
        default=0,
    )
    parser.add_argument(
        "-nv",
        "--no-viewer",
        dest="no_viewer",
        help="Disable viewer",
        action="store_true",
    )
    parser.add_argument(
        "-mp",
        "--multiprocessing",
        help="Whether to run viewer in separate process (default separate thread).",
        dest="multiprocessing",
        action="store_true",
    )
    parser.add_argument(
        "-no-gt",
        "--viewer-disable-gt",
        dest="viewer_disable_gt",
        action="store_true",
        help="Disables display of GT bounding boxes in viewer (and trajectories)",
    )
    parser.add_argument(
        "-no-trajs",
        "--viewer-disable-trajs",
        dest="viewer_disable_trajs",
        action="store_true",
        help="Disables display of trajectories in viewer (both estimated and GT)",
    )
    parser.add_argument(
        "-o",
        "--offset",
        dest="offset",
        help="The image number offset to use (i.e. start at image # <offset> instead of 0)",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--cam",
        dest="cam",
        help="Show objects in camera coordinates in viewer (instead of world)",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--record",
        dest="record",
        help="Record sequence from viewer at given path (ignored if `--no-viewer` is set)",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--continuous",
        action="store_true",
        dest="continuous",
        help="Whether to run process continuously (default is next step via 'n' keypress).",
    )
    parser.add_argument(
        "-t",
        "--tags",
        type=str,
        nargs="+",
        help="One or several tags for the given run (default=`YEAR_MONTH_DAY-HOUR_MINUTE`)",
        default=[datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d-%H_%M")],
    )
    parser.add_argument(
        "--out",
        dest="out",
        type=str,
        help="""Where to save GT and estimated object trajectories
        (default: `<kitti>/trajectories/<scene>/<features>/<tags>[0]/<tags>[1]/.../<tags>[N]`)""",
    )
    args = parser.parse_args()
    LOGGER.setLevel(LOG_LEVELS[args.verbosity])
    scene = str(args.scene).zfill(4)
    if args.multiprocessing:
        queue_class = mp.JoinableQueue
        flag_class = mp.Event
        process_class = mp.Process
    else:
        queue_class = queue.Queue
        flag_class = threading.Event
        process_class = threading.Thread
    shared_data = queue_class()
    returned_data = queue_class()
    slam_data = queue_class()
    stop_flag = flag_class()
    next_step = flag_class()
    kitti_path = Path(config.KITTI_PATH)
    img_shape = get_image_shape(kitti_path, scene)
    image_stream = _get_image_stream(kitti_path, scene, stop_flag, offset=args.offset)
    stereo_cam, T02 = get_cameras_from_kitti(kitti_path)
    gt_poses = get_gt_poses_from_kitti(kitti_path, scene)
    gt_poses = get_gt_poses_from_kitti(kitti_path, scene)
    label_data = get_label_data_from_kitti(
        kitti_path, scene, poses=gt_poses, offset=args.offset
    )
    obj_detections_path = Path(config.DETECTIONS_PATH) / scene
    detection_stream = get_detection_stream(
        obj_detections_path, offset=args.offset, label_data=label_data, object_ids=None,
    )
    disp_process = process_class(
        target=run,
        kwargs={
            "images": image_stream,
            "detections": detection_stream,
            "stereo_cam": stereo_cam,
            "all_poses": gt_poses,
            "shared_data": shared_data,
            "stop_flag": stop_flag,
            "next_step": next_step,
            "returned_data": returned_data,
            "continuous": args.continuous or args.no_viewer,
        },
        name="DISPARITY",
    )
    disp_process.start()

    if args.no_viewer:
        while not stop_flag.is_set():
            shared_data.get(block=True)
            shared_data.task_done()
            next_step.set()
    else:
        run_viewer(
            shared_data=shared_data,
            stop_flag=stop_flag,
            next_step=next_step,
            label_data=label_data,
            save_path=Path(args.record) if args.record else None,
            gt_poses=gt_poses,
            recording=args.record,
            show_trajs=not args.viewer_disable_trajs,
            show_gt=not args.viewer_disable_gt,
            cam_coordinates=args.cam,
        )
    disp_process.join()
    estimated_trajectories_world, estimated_trajectories_cam = returned_data.get()
    returned_data.task_done()
    returned_data.join()
    if not args.out:
        out_path = kitti_path / "trajectories" / scene / config.FEATURE_MATCHER
        for tag in args.tags:
            out_path /= tag
    else:
        out_path = Path(args.out)

    # Save trajectories
    out_path.mkdir(exist_ok=True, parents=True)
    out_est_world = out_path / "est_trajectories_world.json"
    out_est_cam = out_path / "est_trajectories_cam.json"
    # estimated
    with open(out_est_world, "w") as fp:
        json.dump(estimated_trajectories_world, fp, indent=4, sort_keys=True)
    with open(out_est_cam, "w") as fp:
        json.dump(estimated_trajectories_cam, fp, indent=4, sort_keys=True)
    LOGGER.info(
        "Saved estimated object track trajectories to %s", out_path,
    )

    # Save config + git hash
    state_file = out_path / "state.json"
    with open(state_file, "w") as fp:
        state = {}
        state["CONFIG"] = get_config_dict()
        state["HASH"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            encoding="utf-8",
            check=True,
        ).stdout.strip()
        json.dump(state, fp, indent=4)

    # Cleanly shutdown
    while not shared_data.empty():
        shared_data.get()
        shared_data.task_done()
    shared_data.join()
    LOGGER.info("FINISHED RUNNING KITTI GT DISPARITY")
