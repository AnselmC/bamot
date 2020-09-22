import argparse
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _plt_hist(df, dst_dir, save):
    plt.hist(df.error, bins=60, stacked=True, log=True, density=True)
    _, ymax = plt.ylim()
    plt.title("Histogram of errors (60 bins)")
    plt.axvline(
        df.error.mean(),
        linestyle="dashed",
        label=f"Mean: {df.error.mean():.2f}%",
        color="b",
    )
    plt.axvline(
        df.error.median(),
        linestyle="dotted",
        label=f"Median: {df.error.median():.2f}%",
        color="r",
    )
    plt.xlabel("Error")
    plt.ylabel("Count")
    plt.legend()
    if save:
        plt.savefig((dst_dir / "histogram.png").as_posix())
        plt.clf()
    else:
        plt.show()


def _plt_hist_no_outliers(df, dst_dir, save):
    errs = df.where(df.error < df.error.std()).error
    plt.hist(
        errs, bins=60, stacked=True, log=True, density=True,
    )
    _, ymax = plt.ylim()
    plt.title("Histogram of errors < stddev (60 bins)")
    plt.axvline(
        errs.mean(), linestyle="dashed", label=f"Mean: {errs.mean():.2f}%", color="b",
    )
    plt.axvline(
        errs.median(),
        linestyle="dotted",
        label=f"Median: {errs.median():.2f}%",
        color="r",
    )
    plt.xlabel("Error")
    plt.ylabel("Count")
    plt.legend()
    if save:
        plt.savefig((dst_dir / "histogram_no_outliers.png").as_posix())
        plt.clf()
    else:
        plt.show()


def _plt_error_by_dist(df, dst_dir, save):
    plt.title("Distance vs. Error")
    markers = cycle(range(0, 11))
    colors = cycle(plt.cm.Spectral(np.linspace(0, 1, 100)).tolist())
    for scene in df.scene.unique():
        marker = next(markers)
        for obj in df[df.scene == scene]["object id"].unique():
            obj_df = df.where(df.scene == scene).where(df["object id"] == obj)
            plt.scatter(
                obj_df.error,
                obj_df["distance from camera"],
                color=next(colors),
                marker=marker,
                label=f"{scene}, {obj}",
            )
    plt.xlabel("Error [m]")
    plt.ylabel("Distance [m]")
    # plt.legend(loc="upper right")
    if save:
        plt.savefig((dst_dir / f"error_vs_dist.png").as_posix())
        plt.clf()
    else:
        plt.show()


def _write_summary(df, dst_dir, save):
    if save:
        csv_dir = dst_dir / "csv"
        csv_dir.mkdir(exist_ok=True)
        df.to_csv((csv_dir / "full.csv").as_posix())
    summary = ["SUMMARY"]
    df_fully_visible = df.loc[
        (df["truncation level"] == 0) & (df["occlusion level"] == 0)
    ]
    summary.append("*" * 30)
    summary.append(f"Mean error: {df.error.mean()}")
    summary.append(
        f"Mean error (no occlusion or truncation): {df_fully_visible.error.mean()}"
    )
    summary.append(f"Median error: {df.error.median()}")
    summary.append(
        f"Median error (no occlusion or truncation): {df_fully_visible.error.median()}"
    )
    summary.append(f"Standard dev: {df.error.std()}")
    summary.append(
        f"Standard dev (no occlusion or truncation): {df_fully_visible.error.std()}"
    )
    summary.append("-" * 30)
    df_scenes = df.groupby("scene")
    num_scenes = len(df_scenes)
    mean_errors = df_scenes.error.mean()
    median_errors = df_scenes.error.median()
    summary.append(
        f"% scenes w/ mean error < 5m: {(100*(mean_errors < 5).sum()/num_scenes):.2f}%"
    )
    summary.append(
        f"% scenes w/ mean error < 3m: {(100*(mean_errors < 3).sum()/num_scenes):.2f}%"
    )
    summary.append(
        f"% scenes w/ mean error < 1m: {(100*(mean_errors < 1).sum()/num_scenes):.2f}%"
    )
    summary.append(
        f"% scenes w/ median error < 5m: {(100*(median_errors < 5).sum()/num_scenes):.2f}%"
    )
    summary.append(
        f"% scenes w/ median error < 3m: {(100*(median_errors < 3).sum()/num_scenes):.2f}%"
    )
    summary.append(
        f"% scenes w/ median error < 1m: {(100*(median_errors < 1).sum()/num_scenes):.2f}%"
    )
    summary.append("-" * 30)
    df_objs = df.groupby(["scene", "object id"])
    num_objs = len(df_objs)
    mean_error = df_objs.error.mean()
    median_error = df_objs.error.median()
    summary.append(
        f"% objects w/ mean error < 5m: {(100*(mean_error < 5).sum()/num_objs):.2f}%"
    )
    summary.append(
        f"% objects w/ mean error < 3m: {(100*(mean_error < 3).sum()/num_objs):.2f}%"
    )
    summary.append(
        f"% objects w/ mean error < 1m: {(100*(mean_error < 1).sum()/num_objs):.2f}%"
    )
    summary.append(
        f"% objects w/ median error < 5m: {(100*(median_error < 5).sum()/num_objs):.2f}%"
    )
    summary.append(
        f"% objects w/ median error < 3m: {(100*(median_error < 3).sum()/num_objs):.2f}%"
    )
    summary.append(
        f"% objects w/ median error < 1m: {(100*(median_error < 1).sum()/num_objs):.2f}%"
    )
    summary.append("+" * 30)
    summary.append("SCENES")
    for scene in sorted(df.scene.unique()):
        scene_df = df.where(df.scene == scene)
        scene_summary = [f"SCENE {scene}"]
        scene_summary.append(f"Mean error: {scene_df.error.mean()}")
        scene_summary.append(f"Median error: {scene_df.error.median()}")
        scene_summary.append(f"Standard dev: {scene_df.error.std()}")
        num_objs = len(scene_df.groupby("object id"))
        mean_errors = scene_df.groupby("object id").error.mean()
        median_errors = scene_df.groupby("object id").error.median()
        scene_summary.append(
            f"% objects w/ mean error < 5m: {(100*(mean_errors < 5).sum()/num_objs):.2f}%"
        )
        scene_summary.append(
            f"% objects w/ mean error < 3m: {(100*(mean_errors < 3).sum()/num_objs):.2f}%"
        )
        scene_summary.append(
            f"% objects w/ mean error < 1m: {(100*(mean_errors < 1).sum()/num_objs):.2f}%"
        )
        scene_summary.append(
            f"% objects w/ median error < 5m: {(100*(median_errors < 5).sum()/num_objs):.2f}%"
        )
        scene_summary.append(
            f"% objects w/ median error < 3m: {(100*(median_errors < 3).sum()/num_objs):.2f}%"
        )
        scene_summary.append(
            f"% objects w/ median error < 1m: {(100*(median_errors < 1).sum()/num_objs):.2f}%"
        )
        scene_summary.append("+" * 30)
        scene_summary.append("OBJECTS")
        for obj in sorted(df[df.scene == scene]["object id"].unique()):
            obj_summary = [f"TRACK {obj}"]
            obj_df = scene_df.where(df["object id"] == obj)
            obj_summary.append(f"Mean error: {obj_df.error.mean()}")
            obj_summary.append(f"Median error: {obj_df.error.median()}")
            obj_summary.append(f"Standard dev: {obj_df.error.std()}")
            obj_summary.append(
                f"% frames w/ error < 5m: {(100*obj_df.where(obj_df.error < 5).error.count()/obj_df.error.count()):.2f}%"
            )
            obj_summary.append(
                f"% frames w/ error < 3m: {(100*obj_df.where(obj_df.error < 3).error.count()/obj_df.error.count()):.2f}%"
            )
            obj_summary.append(
                f"% frames w/ error < 1m: {(100*obj_df.where(obj_df.error < 1).error.count()/obj_df.error.count()):.2f}%"
            )
            obj_summary.append(
                f"% frames w/ error < 0.5m: {(100*obj_df.where(obj_df.error < 0.5).error.count()/obj_df.error.count()):.2f}%"
            )
            obj_summary.append("." * 30)
            scene_summary.append("\n\t\t".join(obj_summary))
        summary.append("\n\t".join(scene_summary))
    if save:
        with open(dst_dir / "summary.txt", "w") as fp:
            fp.write("\n".join(summary))
    else:
        print("\n".join(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "input",
        help="The evaluation directory containing an `evaluation.csv` file for all scenes",
    )
    parser.add_argument(
        "-s", "--save", dest="save", action="store_true", help="Whether to save plots"
    )

    args = parser.parse_args()
    src_dir = Path(args.input)
    df = None
    for scene_dir in src_dir.iterdir():
        if scene_dir.is_file():
            continue
        eval_file = scene_dir / "evaluation.csv"
        if not eval_file.exists():
            continue
        scene_df = pd.read_csv(eval_file.as_posix())
        scene_df.insert(0, "scene", scene_dir.name)
        if df is None:
            df = scene_df
        else:
            df = pd.concat([df, scene_df], ignore_index=True)
    dst_dir = src_dir / "full_eval"
    dst_dir.mkdir(exist_ok=True)
    # PLOTS
    # histogram of errors
    _write_summary(df, dst_dir, args.save)
    _plt_hist(df, dst_dir, args.save)
    _plt_hist_no_outliers(df, dst_dir, args.save)
    _plt_error_by_dist(df, dst_dir, args.save)
    # histogram of errors w/o outliers
    # errors vs distance
