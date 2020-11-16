import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from bamot.config import CONFIG as config
from bamot.util.kitti import get_gt_poses_from_kitti, get_label_data_from_kitti

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
        "-sc",
        "--scene",
        dest="scene",
        help="the kitti scene",
        choices=range(0, 20),
        type=int,
    )

    parser.add_argument(
        "-s",
        "--save",
        dest="save",
        nargs="?",
        help="Save evaluation and plots (only valid if `--plot` is given, default directory is `data/evaluation/`)",
        const="data/evaluation",
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
    est_trajectory_file = traj_path / "est_trajectories_world.json"
    est_trajectory_file_cam = traj_path / "est_trajectories_cam.json"

    with open(est_trajectory_file.as_posix(), "r") as fp:
        est_trajectories_world = json.load(
            fp,
            object_hook=lambda d: {
                int(k) if k.lstrip("-").isdigit() else k: v for k, v in d.items()
            },
        )

    with open(est_trajectory_file_cam.as_posix(), "r") as fp:
        est_trajectories_cam = json.load(
            fp,
            object_hook=lambda d: {
                int(k) if k.lstrip("-").isdigit() else k: v for k, v in d.items()
            },
        )

    print("Loaded estimated trajectories")
    kitti_path = Path(config.KITTI_PATH)
    if not args.scene:
        # infer scene from dir
        scene = None
        for part in traj_path.as_posix().split("/"):
            try:
                scene = int(part)
                if scene not in range(0, 20):
                    scene = None
            except:
                pass
        if scene is None:
            print(f"Could not infer scene from {kitti_path.as_posix()}")
            exit()
        else:
            args.scene = scene
    scene = str(args.scene).zfill(4)
    gt_poses = get_gt_poses_from_kitti(kitti_path, scene)
    label_data = get_label_data_from_kitti(kitti_path, scene, poses=gt_poses,)
    print("Loaded GT trajectories")

    # if args.plot:
    #    mpl.rcParams["grid.color"] = "w"
    #    mpl.rcParams["grid.color"] = "w"
    #    mpl.rcParams["axes.edgecolor"] = "w"
    #    mpl.rcParams["xtick.color"] = "w"
    #    mpl.rcParams["ytick.color"] = "w"
    if args.save:
        save_dir = Path(args.save) / args.trajectories.split("/")[-1] / scene
        save_dir.mkdir(exist_ok=True, parents=True)
    err_per_obj = {}
    num_objects = args.num_objects if args.num_objects else len(label_data)
    print(f"Number of objects: {num_objects}")
    for j in range(2):
        if args.plot:
            fig_world = plt.figure(figsize=plt.figaspect(0.5))
            ax_3d_world = fig_world.add_subplot(1, 1, 1, projection="3d")
        for i, track_id in enumerate(label_data.keys()):
            if i > num_objects:
                break
            track_dict = label_data[track_id]
            if track_id not in est_trajectories_world:
                continue
            est_traj_dict = est_trajectories_world[track_id]
            est_traj_cam_dict = est_trajectories_cam[track_id]
            err_per_image = {}
            valid_frames = []
            prev_pt = None
            for img_id, row_data in track_dict.items():
                gt_pt_cam = row_data.cam_pos
                gt_pt_cam = np.array(gt_pt_cam).reshape(3, 1)
                est_pt_cam = est_traj_cam_dict.get(img_id)
                gt_pt_world = row_data.world_pos
                gt_pt_world = np.array(gt_pt_world).reshape(3, 1)
                dist = np.linalg.norm(np.array(gt_pt_cam))
                if est_pt_cam is None:
                    tracked = False
                    error = "NA"
                    err_x = "NA"
                    err_y = "NA"
                    err_z = "NA"
                else:
                    tracked = True
                    est_pt_cam = np.array(est_pt_cam).reshape(3, 1)
                    error_cam = np.linalg.norm(gt_pt_cam - est_pt_cam)
                    err_x_cam, err_y_cam, err_z_cam = (
                        np.abs(gt_pt_cam - est_pt_cam).reshape(3,).tolist()
                    )

                    est_pt_world = np.array(est_traj_dict[img_id]).reshape(3, 1)
                    error_world = np.linalg.norm(gt_pt_world - est_pt_world)
                    err_x_world, err_y_world, err_z_world = (
                        np.abs(gt_pt_world - est_pt_world).reshape(3,).tolist()
                    )
                    error = error_cam
                    err_x = err_x_cam
                    err_y = err_y_cam
                    err_z = err_z_cam
                    if args.distances:
                        min_dist, max_dist = map(int, args.distances)
                        if min_dist <= dist <= max_dist:
                            valid_frames.append(img_id)
                    if args.error:
                        max_error = float(args.error)
                        if error < max_error:
                            if img_id not in valid_frames:
                                valid_frames.append(img_id)
                        else:
                            if img_id in valid_frames:
                                valid_frames.remove(img_id)
                err_per_image[img_id] = ((err_x, err_y, err_z, error, tracked), dist)
            if not args.distances and not args.error:
                valid_frames = list(est_traj_dict.keys())
            err_per_obj[track_id] = err_per_image
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
                        ax_3d_world.set_xlabel("x [m]", color="w")
                        ax_3d_world.set_ylabel("y [m]", color="w")
                        ax_3d_world.set_zlabel("z [m]", color="w")
                        # ax_3d.set_yticks([])
                gt_traj_world = np.array(
                    [row.world_pos for row in track_dict.values()]
                ).reshape(-1, 3)
                gt_traj_cam = np.array(
                    [row.cam_pos for row in track_dict.values()]
                ).reshape(-1, 3)
                est_traj_world = np.array(
                    [
                        pt
                        for img_id, pt in est_traj_dict.items()
                        if img_id in valid_frames
                    ]
                ).reshape(-1, 3)
                est_traj_cam = np.array(
                    [
                        pt
                        for img_id, pt in est_traj_cam_dict.items()
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
                # trunaction levels:
                # 0: not truncated
                # 1: partially truncated
                # 2: even more truncated
                truncated_map = {0: 1.0, 1: 0.75, 2: 0.3, 3: 0}
                alphas_occ = [occlusion_map[row.occ_lvl] for row in track_dict.values()]
                alphas_trunc = [
                    truncated_map[row.trunc_lvl] for row in track_dict.values()
                ]
                alphas = [
                    min(occ, trunc) for occ, trunc in zip(alphas_occ, alphas_trunc)
                ]

                colors = np.array(
                    [color.tolist() + [alphas[i]] for i in range(len(gt_traj_world))]
                ).reshape(-1, 4)
                if args.plot in ["both", "trajectory"]:
                    ax_3d.scatter(
                        gt_traj_cam[:, 0],
                        gt_traj_cam[:, 2],
                        gt_traj_cam[:, 1],
                        label="GT trajectory",
                        color=colors,
                        norm=NoNormalize(),
                        linewidths=0.5,
                        marker=".",
                    )
                    ax_3d_world.scatter(
                        gt_traj_world[:, 0],
                        gt_traj_world[:, 1],
                        gt_traj_world[:, 2],
                        label="GT trajectory",
                        color=colors,
                        norm=NoNormalize(),
                        linewidths=0.5,
                        marker=".",
                    )
                    ax_3d.plot(
                        est_traj_cam[:, 0],
                        est_traj_cam[:, 2],
                        est_traj_cam[:, 1],
                        label="Estimated trajectory",
                        color=color,
                    )
                    ax_3d_world.plot(
                        est_traj_world[:, 0],
                        est_traj_world[:, 1],
                        est_traj_world[:, 2],
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
                        err_dist = list(err_per_image.values())
                        err, dist = zip(
                            *[(x[0], x[1]) for x in err_dist if x[0][0] != "NA"]
                        )
                        ax_2d.scatter(
                            [e[0] for e in err],
                            dist,
                            alpha=0.5,
                            label="L1-error x (left, right)",
                        )
                        ax_2d.scatter(
                            [e[1] for e in err],
                            dist,
                            alpha=0.5,
                            label="L1-error y (elevation)",
                        )
                        ax_2d.scatter(
                            [e[2] for e in err],
                            dist,
                            alpha=0.5,
                            label="L1-error z (depth)",
                        )
                        ax_2d.scatter(
                            [e[3] for e in err], dist, label="L1-error total",
                        )
                        # ax_2d.set_xscale("log")
                        ax_2d.set_xlabel("Error [m]")
                        ax_2d.set_ylabel("Distance [m]")
                        plt.legend()
                    if args.plot in ["both", "trajectory"]:
                        ax_3d.view_init(90, -90)
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
                        path = save_dir / f"{track_id}-{args.plot}.pdf"
                        fig.suptitle(f"Object w/ ID {track_id}", color="w")
                        plt.savefig(
                            path.as_posix(), transparent=False, bbox_inches="tight"
                        )
                    else:
                        fig.suptitle(f"Object w/ ID {track_id}")
                        plt.show()

        if j == 1 and args.plot in ["both", "trajectory"]:
            ax_3d_world.view_init(90, -180)
            ax_3d_world.dist = 8
            ax_3d_world.set_xlabel("x [m]", color="w")
            ax_3d_world.set_ylabel("y [m]", color="w")
            ax_3d_world.set_zlabel("z [m]", color="w")
            # ax_3d.xaxis.pane.set_color((0, 0, 0, 0))
            # ax_3d.yaxis.pane.set_color((0, 0, 0, 0))
            # ax_3d.zaxis.pane.set_color((0, 0, 0, 0))
            fig_world.tight_layout()
            # fig.axes[0].axis("off")
            if args.save:
                path = save_dir / "full_view.pdf"
                ax_3d_world.set_yticks([])
                plt.savefig(path.as_posix(), transparent=False, bbox_inches="tight")
            else:
                plt.show()

    mean_err_total = []
    if args.save:
        eval_file = save_dir / "evaluation.csv"
        with open(eval_file.as_posix(), "w") as fp:
            columns = [
                "object_id",
                "image_id",
                "obj_class",
                "distance",
                "occlusion_lvl",
                "truncation_lvl",
                "tracked",
                "error",
                "error_x",
                "error_y",
                "error_z",
            ]
            header = ",".join(columns) + "\n"
            fp.write(header)
            for track_id, err_per_image in err_per_obj.items():
                cls = list(label_data[track_id].values())[0].object_class
                for (
                    img_id,
                    ((err_x_cam, err_y_cam, err_z_cam, error, tracked), dist),
                ) in err_per_image.items():
                    occ_lvl = label_data[track_id][img_id].occ_lvl
                    trunc_lvl = label_data[track_id][img_id].trunc_lvl
                    fp.write(
                        f"{track_id},{img_id},{cls},{dist:.4f},{occ_lvl},{trunc_lvl},{tracked},{error},{err_x_cam},{err_y_cam},{err_z_cam}\n"
                    )
        print(f"Saved results to {eval_file.as_posix()}")
