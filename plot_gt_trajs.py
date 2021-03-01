import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from bamot.util.kitti import (get_cameras_from_kitti,
                              get_gt_detection_data_from_kitti,
                              get_gt_poses_from_kitti)

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
    gt_poses_world = get_gt_poses_from_kitti(kitti_path, scene)
    get_cameras_from_kitti(kitti_path)
    label_data = get_gt_detection_data_from_kitti(
        kitti_path, scene, gt_poses_world, offset=0,
    )
    fig = plt.figure()
    ax_3d = fig.add_subplot(1, 1, 1, projection="3d")
    gt_poses_x, gt_poses_y, gt_poses_z = zip(*(map(lambda x: x[:3, 3], gt_poses_world)))
    ax_3d.plot(gt_poses_x, gt_poses_y, gt_poses_z, linestyle="solid")
    for track_data in label_data.values():
        x, y, z = zip(*list(row.world_pos for row in track_data.values()))
        x = [e[0] for e in x]
        y = [e[0] for e in y]
        z = [e[0] for e in z]
        ax_3d.plot(x, y, z)
    ax_3d.view_init(90, -90)
    ax_3d.set_xlabel("x[m]")
    ax_3d.set_ylabel("y[m]")
    ax_3d.set_zlabel("z[m]")
    plt.show()
