import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import cv2
import g2o
from bamot.util.cv import from_homogeneous_pt, to_homogeneous_pt
from bamot.util.kitti import (get_cameras_from_kitti, get_gt_poses_from_kitti,
                              get_image_stream, get_label_data_from_kitti)


def _get_euler_angles(rot):
    sy = np.sqrt(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rot[2, 1], rot[2, 2])
        y = np.arctan2(-rot[2, 0], sy)
        z = np.arctan2(rot[1, 0], rot[0, 0])
    else:
        x = np.arctan2(-rot[1, 2], rot[1, 1])
        y = np.arctan2(-rot[2, 0], sy)
        z = 0

    return x, y, z


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-k",
        "--kitti-path",
        dest="kitti_path",
        default="./data/KITTI/tracking/training/",
    )
    parser.add_argument(
        "-s",
        "--scene",
        dest="scene",
        help="the scene to plot",
        choices=list(map(str, range(0, 20))),
    )
    args = parser.parse_args()
    scene = args.scene.zfill(4)
    kitti_path = Path(args.kitti_path)
    _, T02 = get_cameras_from_kitti(kitti_path)
    gt_poses_world = get_gt_poses_from_kitti(kitti_path, scene)
    label_data = get_label_data_from_kitti(kitti_path, scene, gt_poses_world)
    ax = plt.axes()
    object_poses_by_img_id = {}
    bounding_boxes_by_img_id = {}
    all_x = []
    all_y = []
    for i in range(len(gt_poses_world)):
        boxes = []
        for track_boxes in label_data.bbox2d.values():
            for img_id, box in track_boxes.items():
                if img_id == i:
                    boxes.append(box)
        bounding_boxes_by_img_id[i] = boxes

        tracks = []
        for track in label_data.world_positions.values():
            if i not in track.keys():
                continue
            obj_positions = []
            for img_id, position in track.items():
                if img_id <= i:
                    x, y, z = position
                    x = x[0]
                    y = y[0]
                    z = z[0]
                    all_x.append(x)
                    all_y.append(y)
                    obj_positions.append([x, y, z])
            if obj_positions:
                tracks.append(obj_positions)
        if tracks:
            object_poses_by_img_id[i] = tracks

    gt_poses_x, gt_poses_y, gt_poses_z = zip(*(map(lambda x: x[:3, 3], gt_poses_world)))
    min_x = min(min(gt_poses_x), min(all_x))
    max_x = max(max(gt_poses_x), max(all_x))
    min_y = min(min(gt_poses_y), min(all_y))
    max_y = max(max(gt_poses_y), max(all_y))
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    for i, (left_img, right_img) in enumerate(get_image_stream(kitti_path, scene)):
        boxes = bounding_boxes_by_img_id.get(i, [])
        if boxes:
            for box in boxes:
                left, top, right, bottom = map(int, box)
                start = (left, top)
                end = (right, bottom)
                color = (255, 0, 0)
                thickness = 2
                left_img = cv2.rectangle(left_img, start, end, color, thickness)
        cv2.imshow("Image", np.vstack([left_img, right_img]))
        cv2.waitKey(1)
        for line in ax.lines:
            line.remove()
        for patch in ax.patches:
            patch.remove()
        current_pose = g2o.Isometry3d(gt_poses_world[i])
        rot = current_pose.R
        arrow = (rot @ np.array([[0, 0, 5]]).T).reshape(3,)

        ax.arrow(current_pose.t[0], current_pose.t[1], arrow[0], arrow[1])
        ax.plot(
            gt_poses_x[: i + 1], gt_poses_y[: i + 1],
        )
        tracks = object_poses_by_img_id.get(i, [])
        for track in tracks:
            x, y, z = zip(*track)
            ax.plot(x, y)
        # plt.legend(loc="lower left")
        plt.draw()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.pause(0.001)

    # plt.plot(gt_poses_x, gt_poses_z, linestyle="solid")
    # plt.show()