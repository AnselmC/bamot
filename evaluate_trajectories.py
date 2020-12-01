import argparse
import json
from pathlib import Path
from typing import NamedTuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import linear_sum_assignment

from bamot.config import CONFIG as config
from bamot.util.kitti import get_gt_poses_from_kitti, get_label_data_from_kitti

RNG = np.random.default_rng()
COLORS = RNG.random((42, 3))


class NoNormalize(mpl.colors.Normalize):
    def __call__(self, value, clip=None):
        return value


class Error(NamedTuple):
    error: float
    error_x: float
    error_y: float
    error_z: float


ErrorUnmatched = Error(np.NAN, np.NAN, np.NAN, np.NAN)


def _get_error(est_pt, gt_pt):
    if est_pt is None:
        return ErrorUnmatched
    else:
        est_pt = np.array(est_pt).reshape(3, 1)
        error = np.linalg.norm(gt_pt - est_pt)
        err_x, err_y, err_z = (
            np.abs(gt_pt - est_pt)
            .reshape(
                3,
            )
            .tolist()
        )
        return Error(error, err_x, err_y, err_z)


def _set_min_max_ax(ax, gt_traj):
    x_min_cam = min(gt_traj[:, 0])
    y_min_cam = min(gt_traj[:, 1])
    z_min_cam = min(gt_traj[:, 2])
    x_max_cam = max(gt_traj[:, 0])
    y_max_cam = max(gt_traj[:, 1])
    z_max_cam = max(gt_traj[:, 2])
    ax.set_xlim(
        x_min_cam - 0.5 * np.abs(x_min_cam), x_max_cam + 0.5 * np.abs(x_max_cam)
    )
    ax.set_ylim(
        y_min_cam - 0.5 * np.abs(y_min_cam), y_max_cam + 0.5 * np.abs(y_max_cam)
    )
    ax.set_zlim(
        z_min_cam - 0.5 * np.abs(z_min_cam), z_max_cam + 0.5 * np.abs(z_max_cam)
    )


def _associate_gt_to_est(est_world, gt_label_data):
    cost_matrix = np.zeros((max(est_world.keys()) + 1, max(gt_label_data.keys()) + 1))
    for track_id_est, est_traj in est_world.items():
        for track_id_gt, track_data in gt_label_data.items():
            # find overlapping img_ids
            overlapping_img_ids = set(est_traj.keys()).intersection(track_data.keys())
            errors = []
            for img_id in overlapping_img_ids:
                est_pt = est_traj[img_id]
                gt_pt = track_data[img_id].world_pos
                error = _get_error(est_pt, gt_pt).error
                if np.isfinite(error):
                    errors.append(error)
            if errors:
                score = 1 / np.median(errors)
            else:
                score = 0
            cost_matrix[track_id_est][track_id_gt] = score
    track_ids_est, track_ids_gt = linear_sum_assignment(cost_matrix, maximize=True)
    track_id_mapping = {}
    matched_est = set()
    matched_gt = set()
    for track_id_est, track_id_gt in zip(track_ids_est, track_ids_gt):
        weight = cost_matrix[track_id_est][track_id_gt].sum()
        if not np.isfinite(weight) or weight == 0:
            continue  # invalid match
        matched_est.add(track_id_est)
        matched_gt.add(track_id_gt)
        track_id_mapping[int(track_id_gt)] = int(track_id_est)

    not_matched_est = set(est_world.keys()).difference(matched_est)
    not_matched_gt = set(gt_label_data.keys()).difference(matched_gt)
    return track_id_mapping, not_matched_gt


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
        "--track-ids-match",
        help="Estimated track ids match the GT track ids",
        action="store_true",
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

    args = parser.parse_args()
    traj_path = Path(args.trajectories)
    offline_path = traj_path / "offline"
    online_path = traj_path / "online"

    if offline_path.exists():  # for compatibility
        est_trajectory_file_offline = offline_path / "est_trajectories_world.json"
        est_trajectory_file_cam_offline = offline_path / "est_trajectories_cam.json"
        est_trajectory_file_online = online_path / "est_trajectories_world.json"
        est_trajectory_file_cam_online = online_path / "est_trajectories_cam.json"
    else:
        est_trajectory_file_offline = traj_path / "est_trajectories_world.json"
        est_trajectory_file_cam_offline = traj_path / "est_trajectories_cam.json"

    with open(est_trajectory_file_offline.as_posix(), "r") as fp:
        est_trajectories_world_offline = json.load(
            fp,
            object_hook=lambda d: {
                int(k) if k.lstrip("-").isdigit() else k: v for k, v in d.items()
            },
        )

    with open(est_trajectory_file_cam_offline.as_posix(), "r") as fp:
        est_trajectories_cam_offline = json.load(
            fp,
            object_hook=lambda d: {
                int(k) if k.lstrip("-").isdigit() else k: v for k, v in d.items()
            },
        )
    # TODO: remove once bug doesn't exist anymore
    if isinstance(est_trajectories_world_offline, list):
        (
            est_trajectories_world_offline,
            est_trajectories_cam_offline,
        ) = est_trajectories_world_offline

    if online_path.exists():
        with open(est_trajectory_file_online.as_posix(), "r") as fp:
            est_trajectories_world_online = json.load(
                fp,
                object_hook=lambda d: {
                    int(k) if k.lstrip("-").isdigit() else k: v for k, v in d.items()
                },
            )

        with open(est_trajectory_file_cam_online.as_posix(), "r") as fp:
            est_trajectories_cam_online = json.load(
                fp,
                object_hook=lambda d: {
                    int(k) if k.lstrip("-").isdigit() else k: v for k, v in d.items()
                },
            )
    else:
        est_trajectories_world_online = {}
        est_trajectories_cam_online = {}

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
    label_data = get_label_data_from_kitti(
        kitti_path,
        scene,
        poses=gt_poses,
    )
    print("Loaded GT trajectories")

    print("Matching GT tracks to estimated tracks")
    if not args.track_ids_match:
        track_mapping, unmatched_gt_track_ids = _associate_gt_to_est(
            est_trajectories_world_offline, label_data
        )
    else:
        track_mapping = dict(zip(label_data.keys(), label_data.keys()))
        unmatched_gt_track_ids = set()
    if args.save:
        save_dir = Path(args.save) / args.trajectories.split("/")[-1] / scene
        save_dir.mkdir(exist_ok=True, parents=True)
    err_per_obj = {}
    num_objects = args.num_objects if args.num_objects else len(label_data)
    print(f"Number of objects: {num_objects}")
    print(f"Number of unmatched GT objects: {len(unmatched_gt_track_ids)}")

    for j in range(2):
        if args.plot:
            fig_world = plt.figure(figsize=plt.figaspect(0.5))
            ax_3d_world = fig_world.add_subplot(1, 1, 1, projection="3d")
        for i, track_id_gt in enumerate(label_data.keys()):
            if i > num_objects:
                break
            if track_id_gt not in track_mapping:
                continue
            track_dict = label_data[track_id_gt]
            track_id_est = track_mapping[track_id_gt]
            est_traj_world_offline_dict = est_trajectories_world_offline[track_id_est]
            est_traj_cam_offline_dict = est_trajectories_cam_offline[track_id_est]
            est_traj_world_online_dict = est_trajectories_world_online.get(
                track_id_est, {}
            )
            est_traj_cam_online_dict = est_trajectories_cam_online.get(track_id_est, {})
            err_per_image = {}
            for img_id, row_data in track_dict.items():
                gt_pt_cam = np.array(row_data.cam_pos).reshape(3, 1)
                est_pt_cam_offline = est_traj_cam_offline_dict.get(img_id)
                est_pt_cam_online = est_traj_cam_online_dict.get(img_id)
                dist_from_camera = np.linalg.norm(np.array(gt_pt_cam))
                error_offline = _get_error(est_pt_cam_offline, gt_pt_cam)
                error_online = _get_error(est_pt_cam_online, gt_pt_cam)

                err_per_image[img_id] = {}
                err_per_image[img_id]["distance"] = dist_from_camera
                err_per_image[img_id]["offline"] = error_offline
                err_per_image[img_id]["online"] = error_online
            err_per_obj[track_id_gt] = err_per_image
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
                gt_traj_world = np.array(
                    [row.world_pos for row in track_dict.values()]
                ).reshape(-1, 3)
                gt_traj_cam = np.array(
                    [row.cam_pos for row in track_dict.values()]
                ).reshape(-1, 3)
                est_traj_world_offline = np.array(
                    list(est_traj_world_offline_dict.values())
                ).reshape(-1, 3)
                est_traj_cam_offline = np.array(
                    list(est_traj_cam_offline_dict.values())
                ).reshape(-1, 3)
                est_traj_world_online = np.array(
                    list(est_traj_world_online_dict.values())
                ).reshape(-1, 3)
                est_traj_cam_online = np.array(
                    list(est_traj_cam_online_dict.values())
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
                    _set_min_max_ax(ax_3d, gt_traj_world)
                    ax_3d.scatter(
                        gt_traj_world[:, 0],
                        gt_traj_world[:, 2],
                        gt_traj_world[:, 1],
                        label="GT trajectory",
                        color=colors,
                        norm=NoNormalize(),
                        linewidths=0.5,
                        marker=".",
                    )
                    # _set_min_max_ax(ax_3d_world, gt_traj_world)
                    ax_3d.plot(
                        est_traj_world_offline[:, 0],
                        est_traj_world_offline[:, 2],
                        est_traj_world_offline[:, 1],
                        label="Estimated offline trajectory",
                        color=color,
                        marker="|",
                    )
                    ax_3d.plot(
                        est_traj_world_online[:, 0],
                        est_traj_world_online[:, 2],
                        est_traj_world_online[:, 1],
                        label="Estimated online trajectory",
                        color=color,
                        marker="+",
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
                    ax_3d_world.plot(
                        est_traj_world_offline[:, 0],
                        est_traj_world_offline[:, 1],
                        est_traj_world_offline[:, 2],
                        label="Estimated offline trajectory",
                        color=color,
                        marker="|",
                    )
                    ax_3d_world.plot(
                        est_traj_world_online[:, 0],
                        est_traj_world_online[:, 1],
                        est_traj_world_online[:, 2],
                        label="Estimated online trajectory",
                        color=color,
                        marker="+",
                    )
                    if j == 0:
                        plt.legend()
                if j == 0:
                    if args.plot in ["both", "error"]:
                        raise NotImplementedError("TODO")
                        if args.plot == "both":
                            ax_2d = fig.add_subplot(1, 2, 2)
                        else:
                            ax_2d = fig.add_subplot(1, 1, 1)
                        err_dist = list(err_per_image.values())
                        err, dist_from_camera = zip(
                            *[(x[0], x[1]) for x in err_dist if x[0][0] != "NA"]
                        )
                        ax_2d.scatter(
                            [e[0] for e in err],
                            dist_from_camera,
                            alpha=0.5,
                            label="L1-error x (left, right)",
                        )
                        ax_2d.scatter(
                            [e[1] for e in err],
                            dist_from_camera,
                            alpha=0.5,
                            label="L1-error y (elevation)",
                        )
                        ax_2d.scatter(
                            [e[2] for e in err],
                            dist_from_camera,
                            alpha=0.5,
                            label="L1-error z (depth)",
                        )
                        ax_2d.scatter(
                            [e[3] for e in err],
                            dist_from_camera,
                            label="L1-error total",
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
                        fig.tight_layout()
                    if args.save:
                        path = save_dir / f"{track_id_gt}-{args.plot}.pdf"
                        fig.suptitle(f"Object w/ GT ID {track_id_gt}", color="w")
                        plt.savefig(
                            path.as_posix(), transparent=False, bbox_inches="tight"
                        )
                    else:
                        fig.suptitle(f"Object w/ GT ID {track_id_gt}")
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
        track_mapping_file = save_dir / "track_mapping.json"
        with open(track_mapping_file, "w") as fp:
            json.dump(track_mapping, fp, sort_keys=True, indent=4)
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
                "error_offline",
                "error_x_offline",
                "error_y_offline",
                "error_z_offline",
                "error_online",
                "error_x_online",
                "error_y_online",
                "error_z_online",
            ]
            header = ",".join(columns) + "\n"
            fp.write(header)
            for track_id, err_per_image in err_per_obj.items():
                cls = list(label_data[track_id].values())[0].object_class
                for img_id, data in err_per_image.items():
                    dist_from_camera = data["distance"]
                    error_offline = data["offline"]
                    error_online = data["online"]
                    tracked = not np.isnan(error_offline.error)
                    occ_lvl = label_data[track_id][img_id].occ_lvl
                    trunc_lvl = label_data[track_id][img_id].trunc_lvl
                    fp.write(
                        (
                            f"{track_id},{img_id},{cls},{dist_from_camera:.4f},{occ_lvl},{trunc_lvl},{tracked},"
                            f"{error_offline.error},{error_offline.error_x},{error_offline.error_y},{error_offline.error_z},"
                            f"{error_online.error},{error_online.error_x},{error_online.error_y},{error_online.error_z}\n"
                        )
                    )
        print(f"Saved results to {eval_file.as_posix()}")
