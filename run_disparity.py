import argparse
import multiprocessing as mp
import queue
import threading
from pathlib import Path

from bamot.config import CONFIG as config
from bamot.core.disparity import run
from bamot.util.kitti import (get_cameras_from_kitti, get_detection_stream,
                              get_gt_poses_from_kitti, get_image_shape,
                              get_label_data_from_kitti)
from bamot.util.viewer import run as run_viewer
from run_kitti_gt_mot import _get_image_stream

if __name__ == "__main__":
    parser = argparse.ArgumentParser("BAMOT with GT KITTI data")
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
    args = parser.parse_args()
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
            "continuous": args.continuous,
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
