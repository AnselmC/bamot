import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("trajectories", help="Path to trajectories to evaluate")
    parser.add_argument(
        "-p", "--plot", dest="plot", action="store_true", help="Turn plotting on"
    )

    args = parser.parse_args()
    traj_path = Path(args.trajectories)
    gt_trajectory_file = traj_path / "gt_trajectories_world.json"
    gt_trajectory_file_cam = traj_path / "gt_trajectories_cam.json"
    est_trajectory_file = traj_path / "est_trajectories_world.json"
    est_trajectory_file_cam = traj_path / "est_trajectories_cam.json"

    with open(gt_trajectory_file.as_posix(), "r") as fp:
        gt_trajectories = json.load(fp)

    with open(gt_trajectory_file_cam.as_posix(), "r") as fp:
        gt_trajectories_cam = json.load(fp)

    with open(est_trajectory_file.as_posix(), "r") as fp:
        est_trajectories = json.load(fp)

    with open(est_trajectory_file_cam.as_posix(), "r") as fp:
        est_trajectories_cam = json.load(fp)

    # per object, get trajectories
    err_dist = {}
    for i, track_id in enumerate(gt_trajectories.keys()):
        gt_traj_dict = gt_trajectories[track_id]
        gt_traj_cam_dict = gt_trajectories_cam[track_id]

        if track_id not in est_trajectories:
            continue
        est_traj_dict = est_trajectories[track_id]
        est_traj_cam_dict = est_trajectories_cam[track_id]
        err_dist_obj = []
        for img_id, est_pt in est_traj_cam_dict.items():
            gt_pt = gt_traj_cam_dict[img_id]
            error = np.linalg.norm(np.array(gt_pt) - np.array(est_pt))
            dist = np.linalg.norm(np.array(gt_pt))
            err_dist_obj.append((error, dist))
        err_dist[track_id] = err_dist_obj
        if args.plot:
            gt_traj = np.array(list(gt_traj_dict.values())).reshape(-1, 3)
            est_traj = np.array(list(est_traj_dict.values())).reshape(-1, 3)
            fig = plt.figure(figsize=plt.figaspect(0.5))
            plt.title(f"Object w/ ID {track_id}")
            ax_3d = fig.add_subplot(1, 2, 1, projection="3d")
            ax_3d.plot(
                gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2], label="GT trajectory",
            )
            ax_3d.plot(
                est_traj[:, 0],
                est_traj[:, 1],
                est_traj[:, 2],
                label="Estimated trajectory",
            )
            ax_3d.set_xlabel("x")
            ax_3d.set_ylabel("y")
            ax_3d.set_zlabel("z")
            plt.legend()
            ax_2d = fig.add_subplot(1, 2, 2)
            ax_2d.scatter(
                list(map(lambda x: x[0], err_dist_obj)),
                list(map(lambda x: x[1], err_dist_obj)),
            )
            ax_2d.set_xlabel("Error [m]")
            ax_2d.set_ylabel("Distance [m]")
            ax_3d.get_shared_x_axes().remove(ax_2d)
            ax_3d.get_shared_y_axes().remove(ax_2d)
            ax_3d.view_init(0, 90)

            fig.tight_layout()
            fig.axes[0].axis("off")
            plt.show()
        # trajectory plot should draw world trajectories
        # error estimates should be made in cam coordinates?
    print("+++EVALUATION+++")
    mean_err_total = []
    for track_id, err_and_dist in err_dist.items():
        print("=" * 40)
        print(f"Object with id {track_id}")
        errors = [err[0] for err in err_and_dist]
        mean_err = np.mean(errors)
        mean_err_total.append(mean_err)
        print(f"Mean error: {np.mean(errors)}")
        print(f"Median error: {np.median(errors)}")
        print(f"Std. dev: {np.std(errors)}")
        print(f"Variance: {np.var(errors)}")
        err_sorted_by_dist = map(
            lambda y: y[0], sorted(err_and_dist, key=lambda x: x[1])
        )
        # print("Errors sorted by distance (from close to far):")
        # print(
        #    *list(err_sorted_by_dist), sep="\n",
        # )
    print(f"Mean error over all tracks: {np.mean(mean_err_total)}")
    print(f"Std. dev over all tracks: {np.std(mean_err_total)}")
    print(f"Variance total: {np.var(mean_err_total)}")
