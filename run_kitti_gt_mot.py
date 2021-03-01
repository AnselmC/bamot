"""Script for BAMOT with GT data from KITTI and mocked SLAM system.
"""
import argparse
import datetime
import glob
import json
import logging
import multiprocessing as mp
import queue
import subprocess
import threading
import time
import warnings
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import colorlog
import cv2
import numpy as np
import tqdm

from bamot.config import CONFIG as config
from bamot.config import get_config_dict
from bamot.core.base_types import StereoImage
from bamot.core.mot import run
from bamot.util.cv import from_homogeneous, to_homogeneous
from bamot.util.kitti import (get_2d_track_line, get_3d_track_line,
                              get_cameras_from_kitti, get_detection_stream,
                              get_gt_detection_data_from_kitti,
                              get_gt_poses_from_kitti, get_image_shape)
from bamot.util.misc import TqdmLoggingHandler
from bamot.util.viewer import run as run_viewer

warnings.filterwarnings(action="ignore")

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


def get_confidence(max_point, value, upward_sloping=True):
    if upward_sloping:
        return 0.5 * (np.tanh(4 * value / max_point - 2) + 1)
    return 0.5 * (np.tanh(2 - 4 * value / max_point) + 1)


def _fake_slam(
    slam_data: Union[queue.Queue, mp.Queue], gt_poses: List[np.ndarray], offset: int
):
    all_poses = []
    for i, pose in enumerate(gt_poses):
        all_poses = gt_poses[: i + 1 + offset]
        slam_data.put(all_poses)
        time.sleep(20 / 1000)  # 20 ms or 50 Hz
    LOGGER.debug("Finished adding fake slam data")


def _write_3d_detections(
    writer_data_3d: Union[queue.Queue, mp.Queue],
    scene: str,
    kitti_path: Path,
    tags: List[str],
):
    path = kitti_path / "3d_tracking"
    for tag in tags:
        path /= tag
    path /= "data"  # to adhere to dir structure expected by ab3dmot
    path.mkdir(parents=True, exist_ok=True)
    fname = path / (scene + ".txt")
    # overwrite file if exists
    if fname.exists():
        fname.unlink()
    while True:
        with open(fname, "a") as fp:
            track_data = writer_data_3d.get(block=True)
            writer_data_3d.task_done()
            if not track_data:
                LOGGER.debug("Finished writing 3d detections")
                break
            img_id = track_data["img_id"]
            LOGGER.debug("Got 3d detection data for image %d", img_id)
            T_world_cam = track_data["T_world_cam"]
            for track_id, track in track_data["tracks"].items():
                obj_type = track.cls.lower()
                dims = config.PED_DIMS if obj_type == "pedestrian" else config.CAR_DIMS
                location = track.locations.get(img_id)
                if location is None:
                    continue
                rot_angle = np.array(track.rot_angle.get(img_id))
                mask = track.masks[0]
                y_top_left, x_top_left = map(min, np.where(mask != 0))
                y_bottom_right, x_bottom_right = map(max, np.where(mask != 0))
                bbox2d = [x_top_left, y_top_left, x_bottom_right, y_bottom_right]
                # beta is angle between z and location dir vector
                loc_cam = from_homogeneous(
                    np.linalg.inv(T_world_cam) @ to_homogeneous(location)
                )
                # kitti locations are given as bottom of bounding box
                # positive y direction is downward, hence add half of the dimensions
                loc_cam[1] += dims[0] / 2
                dir_vec = loc_cam[[0, 2]].reshape(2, 1)
                dir_vec /= np.linalg.norm(dir_vec)
                beta = np.arccos(np.dot(dir_vec.T, np.array([0, 1]).reshape(2, 1)))
                if dir_vec[0] < 0:
                    beta = -beta
                if not np.isfinite(rot_angle):
                    rot_angle = np.array([0])
                alpha = rot_angle - beta

                # number of badly tracked frames negatively effects confidence
                btf_score = get_confidence(
                    max_point=config.KEEP_TRACK_FOR_N_FRAMES_AFTER_LOST,
                    value=track.badly_tracked_frames,
                    upward_sloping=False,
                )

                # number of landmarks positively effects confidence
                num_lm_score = get_confidence(max_point=120, value=len(track.landmarks))

                # number of poses positively effects confidence
                num_pose_score = get_confidence(max_point=10, value=len(track.poses))

                # dist from camera negatively effects confidence
                dist_cam_score = get_confidence(
                    max_point=100, value=track.dist_from_cam, upward_sloping=False,
                )

                confidence_score = (
                    btf_score * num_lm_score * num_pose_score * dist_cam_score
                )

                line = get_3d_track_line(
                    img_id=img_id,
                    track_id=track_id,
                    obj_type=obj_type,
                    dims=dims,
                    loc=loc_cam.flatten().tolist(),
                    rot=rot_angle.flatten()[0],
                    bbox_2d=bbox2d,
                    confidence_score=confidence_score,
                    alpha=alpha.flatten()[0],
                )
                fp.write(line + "\n")


def _write_2d_detections(
    writer_data_2d: Union[queue.Queue, mp.Queue],
    scene: str,
    kitti_path: Path,
    img_shape: Tuple[int, int],
    tags: List[str],
):
    path = kitti_path / "improved_2d_tracking"
    for tag in tags:
        path /= tag
    fname = path / (scene + ".txt")
    height, width = img_shape
    path.mkdir(parents=True, exist_ok=True)
    height, width = img_shape
    with open(fname, "w") as fp:
        while True:
            img_data = writer_data_2d.get(block=True)
            writer_data_2d.task_done()
            if not img_data:
                LOGGER.debug("Finished writing 2d detections")
                break
            img_id = img_data["img_id"]
            LOGGER.debug("Got 2d detection data for image %d", img_id)
            for i in range(len(img_data["track_ids"])):
                track_id = img_data["track_ids"][i]
                mask = img_data["masks"][i]
                cls = img_data["object_classes"][i]
                line = get_2d_track_line(
                    img_id=img_id,
                    track_id=track_id,
                    mask=mask,
                    height=height,
                    width=width,
                    obj_cls=cls,
                )
                fp.write(line + "\n")


def _get_image_stream(
    kitti_path: str,
    scene: str,
    stop_flag: Union[threading.Event, mp.Event],
    offset: int,
) -> Iterable[Tuple[int, StereoImage]]:
    left_img_path = Path(kitti_path) / "image_02" / scene
    right_img_path = Path(kitti_path) / "image_03" / scene
    left_imgs = sorted(glob.glob(left_img_path.as_posix() + "/*.png"))
    right_imgs = sorted(glob.glob(right_img_path.as_posix() + "/*.png"))
    if not left_imgs or not right_imgs or len(left_imgs) != len(right_imgs):
        raise ValueError(
            f"No or differing amount of images found at {left_img_path.as_posix()} and {right_img_path.as_posix()}"
        )

    LOGGER.debug(
        "Found %d left images and %d right images", len(left_imgs), len(right_imgs)
    )
    LOGGER.debug("Starting at image %d", offset)
    img_id = offset
    for left, right in tqdm.tqdm(
        zip(left_imgs[offset:], right_imgs[offset:]), total=len(left_imgs[offset:]),
    ):
        left_img = cv2.imread(left, cv2.IMREAD_COLOR).astype(np.uint8)
        right_img = cv2.imread(right, cv2.IMREAD_COLOR).astype(np.uint8)
        yield img_id, StereoImage(left_img, right_img)
        img_id += 1
    LOGGER.debug("Setting stop flag")
    stop_flag.set()


def _validate_args(args):
    if args.indeces and args.neg_indeces:
        raise ValueError("Can't provide `-id` and `-nid` at the same time")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("BAMOT with GT KITTI data")
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
        "-o",
        "--offset",
        dest="offset",
        help="The image number offset to use (i.e. start at image # <offset> instead of 0)",
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
        "-c",
        "--continuous",
        dest="continuous",
        type=int,
        help=(
            "Up to which image id to run viewer continuously. "
            "If not set, next step can be run via 'n' keypress. "
            "If set without argument, entire sequence is run continuously."
        ),
        nargs="?",
        const=-1,
        default=0,
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
    parser.add_argument(
        "-id",
        "--indeces",
        dest="indeces",
        help="Use this to only track specific object ids (can't be used in conjunction w/ `-nid`)",
        nargs="+",
    )
    parser.add_argument(
        "-nid",
        "--neg-indeces",
        dest="neg_indeces",
        help="Use this to exclude tracks w/ specific object ids (can't be used in conjunction w/ `-id`)",
        nargs="+",
    )
    parser.add_argument(
        "-r",
        "--record",
        dest="record",
        help="Record sequence from viewer at given path (ignored if `--no-viewer` is set)",
        type=str,
    )

    parser.add_argument(
        "-no-gt",
        "--viewer-disable-gt",
        dest="viewer_disable_gt",
        action="store_true",
        help="Disables display of GT bounding boxes in viewer (and trajectories)",
    )
    parser.add_argument(
        "--trajs",
        dest="trajs",
        type=str,
        choices=["offline", "online", "both", "none"],
        help="Which estimated trajectories to display in viewer (`none` disables GT trajectories as well - default: `both`)",
        default="both",
    )
    parser.add_argument(
        "--use-gt",
        help="Use ground truth object masks for left image",
        action="store_true",
    )

    args = parser.parse_args()
    _validate_args(args)
    scene = str(args.scene).zfill(4)
    kitti_path = Path(config.KITTI_PATH)
    if args.use_gt:
        obj_detections_path = Path(config.GT_DETECTIONS_PATH) / scene
    else:
        obj_detections_path = Path(config.EST_DETECTIONS_PATH) / scene

    LOGGER.setLevel(LOG_LEVELS[args.verbosity])

    LOGGER.info(30 * "+")
    LOGGER.info("STARTING BAMOT with GT KITTI data")
    LOGGER.info("SCENE: %s", scene)
    LOGGER.info("USING GT: %s", args.use_gt)
    LOGGER.info("VERBOSITY LEVEL: %s", args.verbosity)
    LOGGER.info("USING MULTI-PROCESS: %s", args.multiprocessing)
    LOGGER.info("CONFIG:\n%s", json.dumps(get_config_dict(), indent=4))
    LOGGER.info(30 * "+")

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
    writer_data_2d = queue_class()
    writer_data_3d = queue_class()
    slam_data = queue_class()
    stop_flag = flag_class()
    next_step = flag_class()
    img_shape = get_image_shape(kitti_path, scene)
    image_stream = _get_image_stream(kitti_path, scene, stop_flag, offset=args.offset)
    stereo_cam, T02 = get_cameras_from_kitti(kitti_path)
    gt_poses = get_gt_poses_from_kitti(kitti_path, scene)
    label_data = get_gt_detection_data_from_kitti(
        kitti_path, scene, poses=gt_poses, offset=args.offset
    )
    detection_stream = get_detection_stream(
        obj_detections_path,
        offset=args.offset,
        label_data=label_data if args.use_gt else None,
        object_ids=[int(idx) for idx in args.indeces] if args.indeces else None,
    )

    slam_process = process_class(
        target=_fake_slam, args=[slam_data, gt_poses, args.offset], name="Fake SLAM"
    )
    write_2d_process = process_class(
        target=_write_2d_detections,
        kwargs={
            "writer_data_2d": writer_data_2d,
            "scene": scene,
            "kitti_path": kitti_path,
            "img_shape": img_shape[:2],
            "tags": args.tags,
        },
        name="2D Detection Writer",
    )
    write_3d_process = process_class(
        target=_write_3d_detections,
        kwargs={
            "writer_data_3d": writer_data_3d,
            "scene": scene,
            "kitti_path": kitti_path,
            "tags": args.tags,
        },
        name="3D Detection Writer",
    )

    continue_until_image_id = -1 if args.no_viewer else args.continuous + args.offset
    mot_process = process_class(
        target=run,
        kwargs={
            "images": image_stream,
            "detections": detection_stream,
            "img_shape": img_shape,
            "stereo_cam": stereo_cam,
            "slam_data": slam_data,
            "shared_data": shared_data,
            "stop_flag": stop_flag,
            "next_step": next_step,
            "returned_data": returned_data,
            "writer_data_2d": writer_data_2d,
            "writer_data_3d": writer_data_3d,
            "continuous_until_img_id": continue_until_image_id,
        },
        name="BAMOT",
    )
    if config.SAVE_UPDATED_2D_TRACK:
        LOGGER.debug("Starting 2d detection writer")
        write_2d_process.start()
    if config.SAVE_3D_TRACK:
        LOGGER.info("Starting 3d detection writer")
        write_3d_process.start()
    LOGGER.debug("Starting fake SLAM")
    slam_process.start()
    LOGGER.debug("Starting MOT")
    mot_process.start()
    LOGGER.debug("Starting viewer")
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
            trajs=args.trajs,
            show_gt=not args.viewer_disable_gt,
            track_ids_match=args.use_gt,
        )
    while not shared_data.empty():
        shared_data.get()
        shared_data.task_done()
        time.sleep(0.5)
    shared_data.join()

    LOGGER.info("No more frames - terminating processes")
    returned = returned_data.get()
    track_id_to_class_mapping = returned["track_id_to_class_mapping"]
    point_cloud_sizes = returned["point_cloud_sizes"]
    offline_trajectories, online_trajectories = returned["trajectories"]
    (
        estimated_trajectories_world_offline,
        estimated_trajectories_cam_offline,
    ) = offline_trajectories
    (
        estimated_trajectories_world_online,
        estimated_trajectories_cam_online,
    ) = online_trajectories

    returned_data.task_done()
    LOGGER.info("Joining returned data queue")
    returned_data.join()
    LOGGER.info("Joined returned data queue")
    LOGGER.info("Joining fake SLAM thread")
    while not slam_data.empty():
        slam_data.get()
        slam_data.task_done()
    slam_data.join()
    slam_process.join()
    LOGGER.info("Joined fake SLAM thread")
    LOGGER.info("Joining MOT thread")
    mot_process.join()
    LOGGER.info("Joined MOT thread")
    if not args.out:
        out_path = kitti_path / "trajectories" / scene / config.FEATURE_MATCHER
        for tag in args.tags:
            out_path /= tag
    else:
        out_path = Path(args.out)

    # Save trajectories
    offline_path = out_path / "offline"
    online_path = out_path / "online"
    offline_path.mkdir(exist_ok=True, parents=True)
    online_path.mkdir(exist_ok=True, parents=True)
    out_est_world_offline = offline_path / "est_trajectories_world.json"
    out_est_cam_offline = offline_path / "est_trajectories_cam.json"
    out_est_world_online = online_path / "est_trajectories_world.json"
    out_est_cam_online = online_path / "est_trajectories_cam.json"
    # estimated (both on- and offline)
    with open(out_est_world_offline, "w") as fp:
        json.dump(estimated_trajectories_world_offline, fp, indent=4, sort_keys=True)
    with open(out_est_cam_offline, "w") as fp:
        json.dump(estimated_trajectories_cam_offline, fp, indent=4, sort_keys=True)
    with open(out_est_world_online, "w") as fp:
        json.dump(estimated_trajectories_world_online, fp, indent=4, sort_keys=True)
    with open(out_est_cam_online, "w") as fp:
        json.dump(estimated_trajectories_cam_online, fp, indent=4, sort_keys=True)
    LOGGER.info(
        "Saved estimated object track trajectories to %s", out_path,
    )
    track_id_to_class_mapping_path = out_path / "track_id_to_class.json"
    with open(track_id_to_class_mapping_path, "w") as fp:
        json.dump(track_id_to_class_mapping, fp, indent=4, sort_keys=True)

    if config.TRACK_POINT_CLOUD_SIZES and point_cloud_sizes:
        point_cloud_size_summary_file = out_path / "pcl.json"
        summary = {}
        max_size = max(map(max, point_cloud_sizes.values()))
        min_size = min(map(min, point_cloud_sizes.values()))
        sum_and_len = [
            (sum(s), len(s)) for s in [sizes for sizes in point_cloud_sizes.values()]
        ]
        total_sum = sum(s[0] for s in sum_and_len)
        total_len = sum(s[1] for s in sum_and_len)
        avg_size = total_sum / total_len
        max_size_obj = max(map(np.mean, point_cloud_sizes.values()))
        min_size_obj = min(map(np.mean, point_cloud_sizes.values()))
        avg_size_obj = np.mean(list(map(np.mean, point_cloud_sizes.values())))
        summary["max"] = max_size
        summary["min"] = min_size
        summary["avg"] = avg_size
        summary["avg-obj"] = avg_size_obj
        summary["min-obj"] = min_size_obj
        summary["max-obj"] = max_size_obj
        LOGGER.info("Point cloud statistics:")
        LOGGER.info(json.dumps(summary, indent=4))
        with open(point_cloud_size_summary_file, "w") as fp:
            json.dump(summary, fp, indent=4)

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

    if config.SAVE_UPDATED_2D_TRACK:
        LOGGER.debug("Joining 2d detection writer")
        write_2d_process.join()
    if config.SAVE_3D_TRACK:
        LOGGER.debug("Joining 3d detection writer")
        write_3d_process.join()
    while not writer_data_2d.empty():
        writer_data_2d.get()
        writer_data_2d.task_done()
        time.sleep(0.5)
    writer_data_2d.join()
    while not writer_data_3d.empty():
        writer_data_3d.get()
        writer_data_3d.task_done()
        time.sleep(0.5)
    writer_data_3d.join()
    LOGGER.info("FINISHED RUNNING KITTI GT MOT")
