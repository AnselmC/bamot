import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

RNG = np.random.default_rng()
COLORS = RNG.random((42, 3))


class NoNormalize(mpl.colors.Normalize):
    def __call__(self, value, clip=None):
        return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("trajectories", help="Path to trajectories to evaluate")
    parser.add_argument(
        "-p",
        "--plot",
        dest="plot",
        nargs="?",
        help="Turn plotting on (default when passed is `both`)",
        choices=["both", "error", "trajectory"],
        const="both",
    )
    parser.add_argument(
        "-s",
        "--save",
        dest="save",
        nargs="?",
        help="Save plots w/ optional path (only valid if `--plot` is given, default directory is `data/plots/`)",
        const="data/plots",
    )

    parser.add_argument(
        "-n", "--num_objects", dest="num_objects", help="Only show the first n objects"
    )
    parser.add_argument(
        "-d",
        "--distances",
        dest="distances",
        nargs=2,
        help="min and max distance of object from camera",
    )
    parser.add_argument("-e", "--error", dest="error", help="max error for plotting")

    args = parser.parse_args()
    traj_path = Path(args.trajectories)
    gt_trajectory_file = traj_path / "gt_trajectories_world.json"
    gt_trajectory_file_cam = traj_path / "gt_trajectories_cam.json"
    est_trajectory_file = traj_path / "est_trajectories_world.json"
    est_trajectory_file_cam = traj_path / "est_trajectories_cam.json"
    occlusion_levels_file = traj_path / "occlusion_levels.json"

    with open(gt_trajectory_file.as_posix(), "r") as fp:
        gt_trajectories = json.load(fp)

    with open(gt_trajectory_file_cam.as_posix(), "r") as fp:
        gt_trajectories_cam = json.load(fp)

    with open(est_trajectory_file.as_posix(), "r") as fp:
        est_trajectories = json.load(fp)

    with open(est_trajectory_file_cam.as_posix(), "r") as fp:
        est_trajectories_cam = json.load(fp)

    if occlusion_levels_file.exists():
        with open(occlusion_levels_file.as_posix(), "r") as fp:
            occlusion_levels = json.load(fp)
    else:
        occlusion_levels = {}

    # per object, get trajectories
    if args.plot and args.save is not None:
        mpl.rcParams["grid.color"] = "w"
        mpl.rcParams["grid.color"] = "w"
        mpl.rcParams["axes.edgecolor"] = "w"
        mpl.rcParams["xtick.color"] = "w"
        mpl.rcParams["ytick.color"] = "w"
        save_dir = Path(args.save) / args.trajectories.split("/")[-1]
        save_dir.mkdir(exist_ok=True, parents=True)
    err_dist = {}
    num_objects = args.num_objects if args.num_objects else len(gt_trajectories)
    for j in range(2):
        if j != 0:
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax_3d = fig.add_subplot(1, 1, 1, projection="3d")
        for i, track_id in enumerate(gt_trajectories.keys()):
            if i > num_objects:
                break
            gt_traj_dict = gt_trajectories[track_id]
            gt_traj_cam_dict = gt_trajectories_cam[track_id]

            if track_id not in est_trajectories:
                continue
            est_traj_dict = est_trajectories[track_id]
            est_traj_cam_dict = est_trajectories_cam[track_id]
            err_dist_obj = []
            valid_frames = []
            for img_id, est_pt in est_traj_cam_dict.items():
                gt_pt = gt_traj_cam_dict.get(img_id)
                if gt_pt is None:  # could be due to occlusion
                    continue
                gt_pt = np.array(gt_pt).reshape(3,).tolist()
                est_pt = np.array(est_pt).reshape(3,).tolist()
                error = np.linalg.norm(np.array(gt_pt) - np.array(est_pt))
                gt_pt_world = np.array(gt_traj_dict[img_id]).reshape(3, 1)
                est_pt_world = np.array(est_traj_dict[img_id]).reshape(3, 1)
                error_world = np.linalg.norm(gt_pt_world - est_pt_world)
                print("error world: ", error_world)
                print("error cam: ", error)
                print("diff: ", np.abs(error_world - error))
                # err_x = np.abs(gt_pt[0] - est_pt[0])
                # err_y = np.abs(gt_pt[1] - est_pt[1])
                # err_z = np.abs(gt_pt[2] - est_pt[2])
                dist = np.linalg.norm(np.array(gt_pt))
                err_x, err_y, err_z = np.abs(gt_pt_world - est_pt_world)
                if args.distances:
                    min_dist, max_dist = map(int, args.distances)
                    if min_dist <= dist <= max_dist:
                        valid_frames.append(img_id)
                if args.error:
                    max_error = float(args.error)
                    if error_world < max_error:
                        if img_id not in valid_frames:
                            valid_frames.append(img_id)
                    else:
                        if img_id in valid_frames:
                            valid_frames.remove(img_id)
                err_dist_obj.append(([err_x, err_y, err_z, error], dist))
            if not args.distances and not args.error:
                valid_frames = list(est_traj_dict.keys())
            err_dist[track_id] = err_dist_obj
            if args.plot:
                if j == 0:
                    fig = plt.figure(figsize=plt.figaspect(0.5))
                    if args.plot == "both":
                        ax_3d = fig.add_subplot(1, 2, 1, projection="3d")
                    elif args.plot == "trajectory":
                        ax_3d = fig.add_subplot(1, 1, 1, projection="3d")
                    if args.plot in ["both", "trajectory"]:
                        ax_3d.set_xlabel("x [m]", color="w")
                        ax_3d.set_ylabel("y [m]", color="w")
                        ax_3d.set_zlabel("z [m]", color="w")
                        # ax_3d.set_yticks([])
                gt_traj = np.array(list(gt_traj_dict.values())).reshape(-1, 3)
                est_traj = np.array(
                    [
                        pt
                        for img_id, pt in est_traj_dict.items()
                        if img_id in valid_frames
                    ]
                ).reshape(-1, 3)
                color = COLORS[RNG.choice(len(COLORS))]
                # occlusion levels:
                # 0: fully visible
                # 1: partly occluded
                # 2: largely occluded
                # 3: unknown
                occlusion_map = {0: 1.0, 1: 0.75, 2: 0.3, 3: 0}
                if occlusion_levels:
                    alphas = list(
                        map(occlusion_map.get, occlusion_levels[track_id].values())
                    )
                else:
                    alphas = [1 for _ in range(len(gt_traj))]
                colors = np.array(
                    [color.tolist() + [alphas[i]] for i in range(len(gt_traj))]
                ).reshape(-1, 4)
                if args.plot in ["both", "trajectory"]:
                    ax_3d.scatter(
                        gt_traj[:, 0],
                        gt_traj[:, 2],
                        gt_traj[:, 1],
                        label="GT trajectory",
                        color=colors,
                        norm=NoNormalize(),
                        linewidths=0.5,
                        marker=".",
                    )
                    ax_3d.plot(
                        est_traj[:, 0],
                        est_traj[:, 2],
                        est_traj[:, 1],
                        label="Estimated trajectory",
                        color=color,
                    )
                    if j == 0:
                        plt.legend()
                if j == 0:
                    if args.plot in ["both", "error"]:
                        if args.plot == "both":
                            ax_2d = fig.add_subplot(1, 2, 2)
                        else:
                            ax_2d = fig.add_subplot(1, 1, 1)
                        ax_2d.scatter(
                            list(map(lambda x: x[0][0], err_dist_obj)),
                            list(map(lambda x: x[1], err_dist_obj)),
                            alpha=0.5,
                            label="L1-error x (left, right)",
                        )
                        ax_2d.scatter(
                            list(map(lambda x: x[0][1], err_dist_obj)),
                            list(map(lambda x: x[1], err_dist_obj)),
                            alpha=0.5,
                            label="L1-error y (elevation)",
                        )
                        ax_2d.scatter(
                            list(map(lambda x: x[0][2], err_dist_obj)),
                            list(map(lambda x: x[1], err_dist_obj)),
                            alpha=0.5,
                            label="L1-error z (depth)",
                        )
                        ax_2d.scatter(
                            list(map(lambda x: x[0][3], err_dist_obj)),
                            list(map(lambda x: x[1], err_dist_obj)),
                            label="L1-error total",
                        )
                        ax_2d.set_xscale("log")
                        ax_2d.set_xlabel("Error [m]")
                        ax_2d.set_ylabel("Distance [m]")
                        plt.legend()
                    if args.plot in ["both", "trajectory"]:
                        ax_3d.view_init(180, 0)
                        if args.plot == "both":
                            ax_3d.get_shared_x_axes().remove(ax_2d)
                            ax_3d.get_shared_y_axes().remove(ax_2d)
                        if args.save:
                            # ax_3d.axis("off")
                            ax_3d.grid(True)
                        ax_3d.xaxis.pane.set_color((1, 1, 1, 0))
                        ax_3d.yaxis.pane.set_color((1, 1, 1, 0))
                        ax_3d.zaxis.pane.set_color((1, 1, 1, 0))
                    if args.save:
                        path = save_dir / f"{track_id}-{args.plot}.png"
                        fig.suptitle(f"Object w/ ID {track_id}", color="w")
                        plt.savefig(
                            path.as_posix(), transparent=True, bbox_inches="tight"
                        )
                    else:
                        fig.suptitle(f"Object w/ ID {track_id}")
                        plt.show()

        if j == 1 and args.plot in ["both", "trajectory"]:
            ax_3d.view_init(180, 0)
            ax_3d.dist = 5
            ax_3d.set_xlabel("x [m]", color="w")
            ax_3d.set_ylabel("y [m]", color="w")
            ax_3d.set_zlabel("z [m]", color="w")
            # ax_3d.xaxis.pane.set_color((0, 0, 0, 0))
            # ax_3d.yaxis.pane.set_color((0, 0, 0, 0))
            # ax_3d.zaxis.pane.set_color((0, 0, 0, 0))
            fig.tight_layout()
            # fig.axes[0].axis("off")
            if args.save:
                path = save_dir / "full_view.png"
                ax_3d.set_yticks([])
                plt.savefig(path.as_posix(), transparent=True, bbox_inches="tight")
            else:
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
