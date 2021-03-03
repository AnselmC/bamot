import argparse
import json
from pathlib import Path
from typing import NamedTuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from g2o import Isometry3d
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import linear_sum_assignment

from bamot.config import CONFIG as config
from bamot.util.kitti import (DetectionData, get_gt_detection_data_from_kitti,
                              get_gt_poses_from_kitti,
                              read_kitti_detection_data)

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
        gt_pt = np.array(gt_pt).reshape(3, 1)
        error = np.linalg.norm(gt_pt - est_pt)
        err_x, err_y, err_z = np.abs(gt_pt - est_pt).reshape(3,).tolist()
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


def associate_gt_to_est(
    est_detection_data: DetectionData, gt_detection_data: DetectionData
):
    cost_matrix = np.zeros((len(est_detection_data), len(gt_detection_data)))
    gt_track_id_mapping = {}
    est_track_id_mapping = {}
    print("GT ids: ", set(gt_detection_data.keys()))
    print("Estimated IDs: ", set(est_detection_data.keys()))
    for i, (track_id_est, est_track_data) in enumerate(est_detection_data.items()):
        for j, (track_id_gt, gt_track_data) in enumerate(gt_detection_data.items()):
            # find overlapping img_ids
            if (
                list(est_track_data.values())[0].object_class.lower()
                != list(gt_track_data.values())[0].object_class.lower()
            ):
                # wrong class
                continue
            overlapping_img_ids = set(est_track_data.keys()).intersection(
                gt_track_data.keys()
            )
            errors = []
            for img_id in overlapping_img_ids:
                est_pt = est_track_data[img_id].world_pos
                gt_pt = gt_track_data[img_id].world_pos
                error = _get_error(est_pt, gt_pt).error
                if np.isfinite(error):
                    errors.append(error)
            if errors:
                score = 1 / np.median(errors) + len(overlapping_img_ids) / len(
                    gt_track_data
                )
            else:
                score = 0
            est_track_id_mapping[i] = track_id_est
            gt_track_id_mapping[j] = track_id_gt
            cost_matrix[i][j] = score
    est_indices, gt_indices = linear_sum_assignment(cost_matrix, maximize=True)
    track_id_mapping = {}
    matched_est = set()
    matched_gt = set()
    for est_idx, gt_idx in zip(est_indices, gt_indices):
        weight = cost_matrix[est_idx][gt_idx].sum()
        est_track_data = est_detection_data[track_id_est]
        gt_track_data = gt_detection_data[track_id_gt]
        overlapping_img_ids = set(est_track_data.keys()).intersection(
            gt_track_data.keys()
        )
        tracked_ratio = len(overlapping_img_ids) / len(gt_track_data.keys())
        if not np.isfinite(weight) or weight == 0:
            continue  # invalid match
        track_id_est = est_track_id_mapping[est_idx]
        track_id_gt = gt_track_id_mapping[gt_idx]
        matched_est.add(track_id_est)
        matched_gt.add(track_id_gt)
        track_id_mapping[int(track_id_gt)] = int(track_id_est)

    not_matched_est = set(est_detection_data.keys()).difference(matched_est)
    not_matched_gt = set(gt_detection_data.keys()).difference(matched_gt)
    print("Not matched est.: ", not_matched_est)
    print("Not matched gt.: ", not_matched_gt)
    return track_id_mapping, not_matched_est, not_matched_gt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "detection_file",
        help="Path to detection file to evaluate (assumes XXXX.txt files in `data` subdirectory).",
    )
    parser.add_argument(
        "-p",
        "--plot",
        dest="plot",
        nargs="?",
        help="Turn plotting on (default when passed is `world`)",
        choices=["both", "world", "individual"],
        const="world",
    )
    parser.add_argument(
        "-sc",
        "--scene",
        dest="scene",
        help="the kitti scene (if not given, will try to determine from `detection_file` path",
        choices=range(0, 21),
        type=int,
    )

    parser.add_argument(
        "-s",
        "--save",
        dest="save",
        nargs="?",
        help="Save evaluation and plots (only valid if `--plot` is given, default directory is `./data/evaluation/`)",
        const="data/evaluation",
    )

    args = parser.parse_args()
    detection_path = Path(args.detection_file)
    kitti_path = Path(config.KITTI_PATH)

    if not args.scene:
        # infer scene from dir
        scene = None
        for part in detection_path.as_posix().split("/"):
            try:
                scene = int(part.split(".")[0])
                if scene not in range(0, 20):
                    scene = None
            except:
                pass
        if scene is None:
            print(f"Could not infer scene from {detection_path.as_posix()}")
            exit()
        else:
            args.scene = scene
    scene = str(args.scene).zfill(4)
    gt_poses = get_gt_poses_from_kitti(kitti_path, scene)
    gt_ego_traj = np.array(
        [Isometry3d(pose).translation() for pose in gt_poses]
    ).reshape(-1, 3)
    gt_detections = get_gt_detection_data_from_kitti(kitti_path, scene, poses=gt_poses,)
    print("Loaded GT detections")
    est_detections = read_kitti_detection_data(detection_path, gt_poses)
    print("Loaded estimated detections")

    print("Matching GT tracks to estimated tracks")
    track_mapping, not_matched_est, not_matched_gt = associate_gt_to_est(
        est_detections, gt_detections
    )

    if args.save:
        save_dir = Path(args.save) / args.trajectories.split("/")[-1] / scene
        save_dir.mkdir(exist_ok=True, parents=True)
    err_per_obj = {}

    if args.plot in ["world", "both"]:
        fig_world = plt.figure(figsize=plt.figaspect(0.5))
        ax_world_gt = fig_world.add_subplot(1, 2, 1)
        ax_world_est = fig_world.add_subplot(
            1, 2, 2, sharex=ax_world_gt, sharey=ax_world_gt
        )
        ax_world_gt.plot(
            gt_ego_traj[:, 0],
            gt_ego_traj[:, 1],
            color="k",
            linewidth=2.5,
            linestyle="dotted",
            label="Ego trajectory",
        )
        ax_world_gt.scatter(
            gt_ego_traj[0, 0], gt_ego_traj[0, 1], marker="o", color="k", label="Start",
        )
        ax_world_gt.scatter(
            gt_ego_traj[-1, 0], gt_ego_traj[-1, 1], marker="x", color="k", label="End",
        )
        ax_world_est.plot(
            gt_ego_traj[:, 0],
            gt_ego_traj[:, 1],
            color="k",
            linewidth=2.5,
            linestyle="dotted",
            label="Ego trajectory",
        )
        ax_world_est.scatter(
            gt_ego_traj[0, 0], gt_ego_traj[0, 1], marker="o", color="k", label="Start",
        )
        ax_world_est.scatter(
            gt_ego_traj[-1, 0], gt_ego_traj[-1, 1], marker="x", color="k", label="End",
        )
        ax_world_est.arrow(
            gt_ego_traj[-2, 0],
            gt_ego_traj[-2, 1],
            gt_ego_traj[-1, 0] - gt_ego_traj[-2, 0],
            gt_ego_traj[-1, 1] - gt_ego_traj[-2, 1],
        )
    for track_id_gt, track_id_est in track_mapping.items():
        # Calculate errors
        gt_track = gt_detections[track_id_gt]
        est_track = est_detections[track_id_est]
        err_per_image = {}
        for img_id, gt_row_data in gt_track.items():
            gt_pt_cam = gt_row_data.cam_pos
            dist_from_camera = np.linalg.norm(np.array(gt_pt_cam))

            est_pt_cam = est_track.get(img_id)
            if est_pt_cam is None:
                continue
            error = _get_error(est_pt_cam.cam_pos, gt_pt_cam)

            err_per_image[img_id] = {}
            err_per_image[img_id]["distance"] = dist_from_camera
            err_per_image[img_id]["error"] = error
        err_per_obj[track_id_gt] = err_per_image

        # plot trajectories and errors
        if args.plot:

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
            alphas_occ = [occlusion_map[row.occ_lvl] for row in gt_track.values()]
            alphas_trunc = [truncated_map[row.trunc_lvl] for row in gt_track.values()]
            alphas = [min(occ, trunc) for occ, trunc in zip(alphas_occ, alphas_trunc)]

            colors = np.array(
                [color.tolist() + [alphas[i]] for i in range(len(gt_track))]
            ).reshape(-1, 4)

            if args.plot in ["both", "individual"]:
                gt_traj_cam = np.array(
                    [row.cam_pos for row in gt_track.values()]
                ).reshape(-1, 3)
                est_traj_cam = np.array(
                    [row.cam_pos for row in est_track.values()]
                ).reshape(-1, 3)
                # fig to hold error and trajectory of object
                track_fig = plt.figure(figsize=plt.figaspect(0.5))

                # trajectory subplot
                ax_3d = track_fig.add_subplot(1, 2, 1, projection="3d")
                # set boundaries of plot to gt boundaries
                _set_min_max_ax(ax_3d, gt_traj_cam)
                ax_3d.set_xlabel("x [m]", color="w")
                ax_3d.set_ylabel("y [m]", color="w")
                ax_3d.set_zlabel("z [m]", color="w")
                ax_3d.scatter(
                    gt_traj_cam[:, 0],
                    gt_traj_cam[:, 2],
                    gt_traj_cam[:, 1],
                    label="GT trajectory in cam. coordinates",
                    color=colors,
                    norm=NoNormalize(),
                    linewidths=0.5,
                    marker=".",
                )
                ax_3d.plot(
                    est_traj_cam[:, 0],
                    est_traj_cam[:, 2],
                    est_traj_cam[:, 1],
                    label="Est. trajectory in cam. coordinates",
                    color=color,
                    marker="|",
                )
                # error subplot
                ax_2d = track_fig.add_subplot(1, 2, 2)
                err_dist = list(err_per_image.values())
                err = [x["error"] for x in err_dist if x["error"] != "NA"]
                dist_from_camera = [
                    x["distance"] for x in err_dist if x["error"] != "NA"
                ]
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
                    [e[3] for e in err], dist_from_camera, label="L1-error total",
                )
                # ax_2d.set_xscale("log")
                ax_2d.set_xlabel("Error [m]")
                ax_2d.set_ylabel("Distance [m]")
                ax_3d.view_init(90, -90)
                ax_3d.get_shared_x_axes().remove(ax_2d)
                ax_3d.get_shared_y_axes().remove(ax_2d)
                ax_3d.xaxis.pane.set_color((1, 1, 1, 0))
                ax_3d.yaxis.pane.set_color((1, 1, 1, 0))
                ax_3d.zaxis.pane.set_color((1, 1, 1, 0))
                track_fig.tight_layout()
                if args.save:
                    path = save_dir / f"{track_id_gt}-{args.plot}.pdf"
                    track_fig.suptitle(f"Object w/ GT ID {track_id_gt}", color="w")
                    plt.savefig(path.as_posix(), transparent=False, bbox_inches="tight")
                    # ax_3d.axis("off")
                    ax_3d.grid(True)
                else:
                    track_fig.suptitle(f"Object w/ GT ID {track_id_gt}")

            if args.plot in ["both", "world"]:
                gt_traj_world = np.array(
                    [row.world_pos for row in gt_track.values()]
                ).reshape(-1, 3)

                est_traj_world = np.array(
                    [row.world_pos for row in est_track.values()]
                ).reshape(-1, 3)
                # _set_min_max_ax(ax_3d_world, gt_traj_world)
                ax_world_gt.plot(
                    gt_traj_world[:, 0],
                    gt_traj_world[:, 1],
                    color=color,
                    # norm=NoNormalize(),
                    linewidth=1,
                )
                ax_world_est.plot(
                    est_traj_world[:, 0], est_traj_world[:, 1], color=color, linewidth=1
                )

    if args.plot in ["both", "world"]:
        fig_world.suptitle(f"Scene {scene}")
        ax_world_gt.set_title("Ground truth")
        ax_world_gt.set_xlabel("x [m]")
        ax_world_gt.set_ylabel("y [m]")
        ax_world_gt.legend()

        ax_world_est.set_title("Estimated")
        ax_world_est.set_xlabel("x [m]")
        ax_world_est.set_ylabel("y [m]")
        ax_world_est.legend()

        ax_world_gt.spines["top"].set_visible(False)
        ax_world_gt.spines["right"].set_visible(False)
        ax_world_est.spines["top"].set_visible(False)
        ax_world_est.spines["right"].set_visible(False)

        fig_world.tight_layout()
        if args.save:
            path = save_dir / "full_view.pdf"
            ax_world_gt.set_yticks([])
            plt.savefig(path.as_posix(), transparent=False, bbox_inches="tight")
    if args.plot and not args.save:
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
                cls = list(gt_detections[track_id].values())[0].object_class
                for img_id, data in err_per_image.items():
                    dist_from_camera = data["distance"]
                    error_offline = data["offline"]
                    error = data["online"]
                    tracked = not np.isnan(error_offline.error)
                    occ_lvl = gt_detections[track_id][img_id].occ_lvl
                    trunc_lvl = gt_detections[track_id][img_id].trunc_lvl
                    fp.write(
                        (
                            f"{track_id},{img_id},{cls},{dist_from_camera:.4f},{occ_lvl},{trunc_lvl},{tracked},"
                            f"{error_offline.error},{error_offline.error_x},{error_offline.error_y},{error_offline.error_z},"
                            f"{error.error},{error.error_x},{error.error_y},{error.error_z}\n"
                        )
                    )
        print(f"Saved results to {eval_file.as_posix()}")
