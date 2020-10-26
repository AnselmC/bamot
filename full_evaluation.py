import argparse
from itertools import cycle
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def _plt_hist(df, dst_dir, save):
    print("Plotting histogram...")
    plt.hist(df.error, bins=60, stacked=True, log=True, density=True)
    _, ymax = plt.ylim()
    plt.title("Histogram of errors (60 bins)")
    plt.axvline(
        df.error.mean(),
        linestyle="dashed",
        label=f"Mean: {df.error.mean():.2f}",
        color="b",
    )
    plt.axvline(
        df.error.median(),
        linestyle="dotted",
        label=f"Median: {df.error.median():.2f}",
        color="r",
    )
    plt.xlabel("Error")
    plt.ylabel("Count")
    plt.legend()
    if save:
        plt.savefig((dst_dir / "histogram.pdf").as_posix())
        plt.clf()
    else:
        plt.show()
    print("Done!")


def _plt_hist_no_outliers(df, dst_dir, save):
    print("Plotting histogram w/o outliers...")
    errs = df.where(df.error < df.error.std()).error
    plt.hist(
        errs, bins=60, stacked=True, log=True, density=True,
    )
    _, ymax = plt.ylim()
    plt.title("Histogram of errors < stddev (60 bins)")
    plt.axvline(
        errs.mean(), linestyle="dashed", label=f"Mean: {errs.mean():.2f}", color="b",
    )
    plt.axvline(
        errs.median(),
        linestyle="dotted",
        label=f"Median: {errs.median():.2f}",
        color="r",
    )
    plt.xlabel("Error")
    plt.ylabel("Count")
    plt.legend()
    if save:
        plt.savefig((dst_dir / "histogram_no_outliers.pdf").as_posix())
        plt.clf()
    else:
        plt.show()
    print("Done!")


def _plt_error_by_dist(df, dst_dir, save):
    print("Plotting error by distance...")
    plt.title("Distance vs. Error")
    markers = cycle(range(0, 11))
    colors = cycle(plt.cm.Spectral(np.linspace(0, 1, 100)).tolist())
    for scene in df.scene.unique():
        marker = next(markers)
        for obj in df[df.scene == scene].object_id.unique():
            obj_df = df.where(df.scene == scene).where(df.object_id == obj)
            plt.scatter(
                obj_df.error,
                obj_df.distance,
                color=next(colors),
                marker=marker,
                label=f"{scene}, {obj}",
            )
    plt.xlabel("Error [m]")
    plt.ylabel("Distance [m]")
    # plt.legend(loc="upper right")
    if save:
        plt.savefig((dst_dir / "error_vs_dist.pdf").as_posix())
        plt.clf()
    else:
        plt.show()
    print("Done")


def _summarize_df_by_dist(df):
    dist_summary = {}
    df_close = df.loc[df.distance < 5]
    df_mid = df.loc[(df.distance >= 5) & (df.distance < 30)]
    df_far = df.loc[df.distance >= 30]
    dist_summary["close"] = _get_metrics(df_close)
    dist_summary["mid"] = _get_metrics(df_mid)
    dist_summary["far"] = _get_metrics(df_far)
    return dist_summary


def _summarize_df(df):
    summary = {}
    summary["total"] = _get_metrics(df)
    df_fully_visible = df.loc[(df.truncation_lvl == 0) & (df.occlusion_lvl == 0)]
    summary["fully-visible"] = _get_metrics(df_fully_visible)
    obj_summary = {}
    df_ped = df.loc[df.obj_class == "Pedestrian"]
    df_car = df.loc[df.obj_class == "Car"]
    obj_summary["car"] = _get_metrics(df_car)
    obj_summary["pedestrian"] = _get_metrics(df_ped)
    summary["obj-type"] = obj_summary
    summary["distance"] = _summarize_df_by_dist(df)
    summary["per-obj"] = _summarize_per_obj(df)
    return summary


def _summarize_best_worst(df, group, summarize_item):
    df_groupedby = df.groupby(group)
    mean_errors = df_groupedby.error.mean()
    median_errors = df_groupedby.error.median()
    summary = {}
    best = {}
    best_mean = []
    for item in mean_errors.nsmallest(n=10).items():
        best_mean.append(summarize_item(df, item))
    best_median = []
    for item in median_errors.nsmallest(n=10).items():
        best_median.append(summarize_item(df, item))

    worst = {}
    worst_mean = []
    for item in mean_errors.nlargest(n=10).items():
        worst_mean.append(summarize_item(df, item))
    worst_median = []
    for item in median_errors.nlargest(n=10).items():
        worst_median.append(summarize_item(df, item))

    best["median"] = best_median
    best["mean"] = best_mean
    worst["median"] = worst_median
    worst["mean"] = worst_mean
    summary["worst"] = worst
    summary["best"] = best
    return summary


def _summarize_per_obj(df):
    df_groupedby_obj = df.groupby(["scene", "object_id", "obj_class"])
    num_objs = len(df_groupedby_obj)
    mean_errors = df_groupedby_obj.error.mean()
    median_errors = df_groupedby_obj.error.median()
    obj_summary = {}
    obj_summary["mean-of-mean"] = float(mean_errors.mean())
    obj_summary["median-of-mean"] = float(mean_errors.median())
    obj_summary["mean-of-median"] = float(median_errors.mean())
    obj_summary["median-of-median"] = float(median_errors.median())
    obj_summary["mean-error-lt-10-pct"] = float(
        100 * (mean_errors < 10).sum() / num_objs
    )
    obj_summary["mean-error-lt-5-pct"] = float(100 * (mean_errors < 5).sum() / num_objs)
    obj_summary["mean-error-lt-3-pct"] = float(100 * (mean_errors < 3).sum() / num_objs)
    obj_summary["mean-error-lt-1-pct"] = float(100 * (mean_errors < 1).sum() / num_objs)
    obj_summary["median-error-lt-10-pct"] = float(
        100 * (median_errors < 10).sum() / num_objs
    )
    obj_summary["median-error-lt-5-pct"] = float(
        100 * (median_errors < 5).sum() / num_objs
    )
    obj_summary["median-error-lt-3-pct"] = float(
        100 * (median_errors < 3).sum() / num_objs
    )
    obj_summary["median-error-lt-1-pct"] = float(
        100 * (median_errors < 1).sum() / num_objs
    )
    best_worst = _summarize_best_worst(
        df, group=["scene", "object_id", "obj_class"], summarize_item=_get_obj_summary
    )
    obj_summary.update(best_worst)
    return obj_summary


def _get_scene_summary(df, scene_row):
    scene, error = scene_row
    df_scene = df[df.scene == scene]
    num_frames_total = len(df_scene)
    num_objects = len(df_scene.object_id.unique())
    max_error = float(df_scene.error.max())
    min_error = float(df_scene.error.min())
    return {
        "scene": scene,
        "error": error,
        "num_frames": num_frames_total,
        "num_objects": num_objects,
        "max_error": max_error,
        "min_error": min_error,
    }


def _get_obj_summary(df, obj):
    scene, obj_id, obj_class = obj[0]
    error = obj[1]
    df_obj = df.loc[(df.scene == scene) & (df.object_id == obj_id)]
    num_frames_total = len(df_obj)
    try:
        tracked_ratio = float(df_obj.tracked.sum() / num_frames_total)
    except AttributeError:
        tracked_ratio = "NA"
    df_fully_visible = df_obj.loc[(df.truncation_lvl == 0) & (df.occlusion_lvl == 0)]
    fully_visible_ratio = len(df_fully_visible) / num_frames_total
    min_dist_from_cam = float(df_obj.distance.min())
    max_dist_from_cam = float(df_obj.distance.max())
    return {
        "scene": scene,
        "object_id": obj_id,
        "class": obj_class,
        "error": error,
        "num_frames": num_frames_total,
        "tracked_ratio": tracked_ratio,
        "fully_visible_ratio": fully_visible_ratio,
        "min_dist": min_dist_from_cam,
        "max_dist": max_dist_from_cam,
    }


def _get_metrics(df):
    metrics = {}
    metrics["mean"] = float(df.error.mean())
    metrics["median"] = float(df.error.median())
    metrics["stddev"] = float(df.error.std())
    if df.empty:
        metrics["tracked_ratio"] = "NA"
    else:
        try:
            metrics["tracked_ratio"] = float(df.tracked.sum() / len(df))
        except AttributeError:
            metrics["tracked_ratio"] = "NA"
    return metrics


def _create_report(df, dst_dir, save):
    if save:
        csv_dir = dst_dir / "csv"
        csv_dir.mkdir(exist_ok=True)
        df.to_csv((csv_dir / "full.csv").as_posix())
    report = {}
    print("Creating overall summary...")
    report["summary"] = _summarize_df(df)
    print("Created!")
    per_scene_report = {}
    report["per-scene"] = _summarize_best_worst(
        df, group=["scene"], summarize_item=_get_scene_summary
    )
    for scene in sorted(df.scene.unique()):
        print(f"Creating summary for scene {scene}")
        scene_df = df[df.scene == scene]
        per_scene_report[scene] = _summarize_df(scene_df)
        print("Created!")
    report["per-scene"].update(per_scene_report)

    if save:
        print("Saving report")
        with open(dst_dir / "report.yaml", "w") as fp:
            yaml.dump(report, fp, sort_keys=False)
    else:
        pprint(report)
    print("Finished report!")


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
        # for backward compatibility
        scene_df.rename(
            columns={
                "occlusion level": "occlusion_lvl",
                "truncation level": "truncation_lvl",
                "object id": "object_id",
                "image id": "image_id",
                "distance from camera": "distance",
            },
            inplace=True,
        )

        scene_df.insert(0, "scene", scene_dir.name)
        if df is None:
            df = scene_df
        else:
            df = pd.concat([df, scene_df], ignore_index=True)
    dst_dir = src_dir / "full_eval"
    dst_dir.mkdir(exist_ok=True)
    # PLOTS
    # histogram of errors
    _create_report(df, dst_dir, args.save)
    _plt_hist(df, dst_dir, args.save)
    _plt_hist_no_outliers(df, dst_dir, args.save)
    # _plt_error_by_dist(df, dst_dir, args.save)
    # histogram of errors w/o outliers
    # errors vs distance
