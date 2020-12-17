import argparse
import json
import warnings
from itertools import cycle
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from evaluate_trajectories import MOTMetrics

warnings.filterwarnings("ignore")


def _plt_hist(df, dst_dir, save):
    print("Plotting histogram...")
    plt.hist(df.error_offline, bins=60, stacked=True, log=True, density=True)
    _, ymax = plt.ylim()
    plt.title("Histogram of errors (60 bins)")
    plt.axvline(
        df.error_offline.mean(),
        linestyle="dashed",
        label=f"Mean: {df.error_offline.mean():.2f}",
        color="b",
    )
    plt.axvline(
        df.error_offline.median(),
        linestyle="dotted",
        label=f"Median: {df.error_offline.median():.2f}",
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
    errs = df.where(df.error_offline < df.error_offline.std()).error_offline
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
    print("Plotting error_offline by distance...")
    plt.title("Distance vs. Error")
    markers = cycle(range(0, 11))
    colors = cycle(plt.cm.Spectral(np.linspace(0, 1, 100)).tolist())
    for scene in df.scene.unique():
        marker = next(markers)
        for obj in df[df.scene == scene].object_id.unique():
            obj_df = df.where(df.scene == scene).where(df.object_id == obj)
            plt.scatter(
                obj_df.error_offline,
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
    obj_summary["car"].update(_summarize_per_obj(df_car))
    obj_summary["pedestrian"] = _get_metrics(df_ped)
    obj_summary["pedestrian"].update(_summarize_per_obj(df_ped))
    summary["obj-type"] = obj_summary
    summary["distance"] = _summarize_df_by_dist(df)
    summary["per-obj"] = _summarize_per_obj(df, summarize_best_worst=True)
    return summary


def _summarize_best_worst(df, group, summarize_item):
    df_groupedby = df.groupby(group)
    mean_errors_offline = df_groupedby.error_offline.mean()
    median_errors_offline = df_groupedby.error_offline.median()
    mean_errors_online = df_groupedby.error_online.mean()
    median_errors_online = df_groupedby.error_online.median()
    summary = {}
    best = {}
    best_mean_offline = []
    best_mean_online = []
    for item in mean_errors_offline.nsmallest(n=3).items():
        best_mean_offline.append(summarize_item(df, item))
    for item in mean_errors_online.nsmallest(n=3).items():
        best_mean_online.append(summarize_item(df, item))
    best_median_offline = []
    best_median_online = []
    for item in median_errors_offline.nsmallest(n=3).items():
        best_median_offline.append(summarize_item(df, item))
    for item in median_errors_online.nsmallest(n=3).items():
        best_median_online.append(summarize_item(df, item))

    worst = {}
    worst_mean_offline = []
    worst_mean_online = []
    for item in mean_errors_offline.nlargest(n=3).items():
        worst_mean_offline.append(summarize_item(df, item))
    for item in mean_errors_online.nlargest(n=3).items():
        worst_mean_online.append(summarize_item(df, item))
    worst_median_offline = []
    worst_median_online = []
    for item in median_errors_offline.nlargest(n=3).items():
        worst_median_offline.append(summarize_item(df, item))
    for item in median_errors_online.nlargest(n=3).items():
        worst_median_online.append(summarize_item(df, item))

    best_offline = {}
    best_online = {}
    best_offline["median"] = best_median_offline
    best_offline["mean"] = best_mean_offline
    best_online["median"] = best_median_online
    best_online["mean"] = best_mean_online
    best["offline"] = best_offline
    best["online"] = best_online

    worst_offline = {}
    worst_online = {}
    worst_offline["median"] = worst_median_offline
    worst_offline["mean"] = worst_mean_offline
    worst_online["median"] = worst_median_online
    worst_online["mean"] = worst_mean_online
    worst["offline"] = worst_offline
    worst["online"] = worst_online
    summary["worst"] = worst
    summary["best"] = best
    return summary


def _summarize_per_obj(df, summarize_best_worst=False):
    obj_summary = {}
    df_groupedby_obj = df.groupby(["scene", "object_id", "obj_class"])
    obj_summary["all"] = _summarize_groupedby_obj(
        df_groupedby_obj, summarize_best_worst=summarize_best_worst
    )
    df_groupedby_obj_close = df.loc[
        df_groupedby_obj.distance.transform("mean") < 5
    ].groupby(["scene", "object_id", "obj_class"])
    obj_summary["close"] = _summarize_groupedby_obj(df_groupedby_obj_close)
    df_groupedby_obj_mid = df.loc[
        df_groupedby_obj.distance.transform("mean") < 30
    ].groupby(["scene", "object_id", "obj_class"])
    obj_summary["mid"] = _summarize_groupedby_obj(df_groupedby_obj_mid)
    df_groupedby_obj_far = df.loc[
        df_groupedby_obj.distance.transform("mean") > 30
    ].groupby(["scene", "object_id", "obj_class"])
    obj_summary["far"] = _summarize_groupedby_obj(df_groupedby_obj_far)
    # outliers: close: num_objects * median error_offline gt 1
    #           mid: num_objects * median error_offline gt 5
    #           far: num_objects * median error_offline gt 10
    # divided by num objects from all
    num_outliers_close_offline = (
        obj_summary["close"]["num-objs"]
        - obj_summary["close"]["offline"]["median-error-lt-1"]
    )
    num_outliers_mid_offline = (
        obj_summary["mid"]["num-objs"]
        - obj_summary["mid"]["offline"]["median-error-lt-5"]
    )
    num_outliers_far_offline = (
        obj_summary["far"]["num-objs"]
        - obj_summary["far"]["offline"]["median-error-lt-10"]
    )
    num_outliers_offline = (
        num_outliers_close_offline + num_outliers_mid_offline + num_outliers_far_offline
    )
    num_outliers_close_online = (
        obj_summary["close"]["num-objs"]
        - obj_summary["close"]["online"]["median-error-lt-1"]
    )
    num_outliers_mid_online = (
        obj_summary["mid"]["num-objs"]
        - obj_summary["mid"]["online"]["median-error-lt-5"]
    )
    num_outliers_far_online = (
        obj_summary["far"]["num-objs"]
        - obj_summary["far"]["online"]["median-error-lt-10"]
    )
    num_outliers_online = (
        num_outliers_close_online + num_outliers_mid_online + num_outliers_far_online
    )
    num_objs = obj_summary["all"]["num-objs"]
    if num_objs:
        obj_summary["outlier-ratio"] = {}
        obj_summary["outlier-ratio"]["offline"] = min(
            num_outliers_offline / num_objs, 1
        )
        obj_summary["outlier-ratio"]["online"] = min(num_outliers_online / num_objs, 1)
    else:
        obj_summary["outlier-ratio"] = "NA"
    return obj_summary


def _summarize_groupedby_obj(df_groupedby_obj, summarize_best_worst=False):
    num_objs = len(df_groupedby_obj)
    mean_errors_offline = df_groupedby_obj.error_offline.mean()
    median_errors_offline = df_groupedby_obj.error_offline.median()
    mean_errors_online = df_groupedby_obj.error_online.mean()
    median_errors_online = df_groupedby_obj.error_online.median()
    obj_summary = {}
    obj_summary["num-objs"] = num_objs
    offline_summary = {}
    offline_summary["mean-of-mean"] = float(mean_errors_offline.mean())
    offline_summary["median-of-mean"] = float(mean_errors_offline.median())
    offline_summary["mean-of-median"] = float(median_errors_offline.mean())
    offline_summary["median-of-median"] = float(median_errors_offline.median())
    offline_summary["mean-error-lt-10"] = float((mean_errors_offline < 10).sum())
    offline_summary["mean-error-lt-5"] = float((mean_errors_offline < 5).sum())
    offline_summary["mean-error-lt-3"] = float((mean_errors_offline < 3).sum())
    offline_summary["mean-error-lt-1"] = float((mean_errors_offline < 1).sum())
    offline_summary["median-error-lt-10"] = float((median_errors_offline < 10).sum())
    offline_summary["median-error-lt-5"] = float((median_errors_offline < 5).sum())
    offline_summary["median-error-lt-3"] = float((median_errors_offline < 3).sum())
    offline_summary["median-error-lt-1"] = float((median_errors_offline < 1).sum())
    obj_summary["offline"] = offline_summary
    online_summary = {}
    online_summary["mean-of-mean"] = float(mean_errors_online.mean())
    online_summary["median-of-mean"] = float(mean_errors_online.median())
    online_summary["mean-of-median"] = float(median_errors_online.mean())
    online_summary["median-of-median"] = float(median_errors_online.median())
    online_summary["mean-error-lt-10"] = float((mean_errors_online < 10).sum())
    online_summary["mean-error-lt-5"] = float((mean_errors_online < 5).sum())
    online_summary["mean-error-lt-3"] = float((mean_errors_online < 3).sum())
    online_summary["mean-error-lt-1"] = float((mean_errors_online < 1).sum())
    online_summary["median-error-lt-10"] = float((median_errors_online < 10).sum())
    online_summary["median-error-lt-5"] = float((median_errors_online < 5).sum())
    online_summary["median-error-lt-3"] = float((median_errors_online < 3).sum())
    online_summary["median-error-lt-1"] = float((median_errors_online < 1).sum())
    obj_summary["online"] = online_summary

    if summarize_best_worst:
        best_worst = _summarize_best_worst(
            df,
            group=["scene", "object_id", "obj_class"],
            summarize_item=_get_obj_summary,
        )
        obj_summary.update(best_worst)
    return obj_summary


def _get_scene_summary(df, scene_row):
    scene, error_offline = scene_row
    df_scene = df[df.scene == scene]
    num_frames_total = len(df_scene)
    num_objects = len(df_scene.object_id.unique())
    max_error_offline = float(df_scene.error_offline.max())
    min_error_offline = float(df_scene.error_offline.min())
    max_error_online = float(df_scene.error_online.max())
    min_error_online = float(df_scene.error_online.min())
    return {
        "scene": scene,
        "error_offline": error_offline,
        "num_frames": num_frames_total,
        "num_objects": num_objects,
        "max_error_offline": max_error_offline,
        "min_error_offline": min_error_offline,
        "max_error_online": max_error_online,
        "min_error_online": min_error_online,
    }


def _get_obj_summary(df, obj):
    scene, obj_id, obj_class = obj[0]
    error_offline = obj[1]
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
        "error_offline": error_offline,
        "num_frames": num_frames_total,
        "tracked_ratio": tracked_ratio,
        "fully_visible_ratio": fully_visible_ratio,
        "min_dist": min_dist_from_cam,
        "max_dist": max_dist_from_cam,
    }


def _get_metrics(df):
    metrics = {}
    errors_offline = {}
    errors_offline["mean"] = float(df.error_offline.mean())
    errors_offline["median"] = float(df.error_offline.median())
    errors_offline["stddev"] = float(df.error_offline.std())
    errors_online = {}
    errors_online["mean"] = float(df.error_online.mean())
    errors_online["median"] = float(df.error_online.median())
    errors_online["stddev"] = float(df.error_online.std())
    metrics["offline"] = errors_offline
    metrics["online"] = errors_online
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
    full_mot_metrics = MOTMetrics()
    for scene_dir in src_dir.iterdir():
        if scene_dir.is_file():
            continue
        eval_file = scene_dir / "evaluation.csv"
        if not eval_file.exists():
            continue
        mot_metrics_file = scene_dir / "mot_metrics.json"
        with open(mot_metrics_file, "r") as fp:
            scene_mot_metrics = MOTMetrics(**json.load(fp))
        full_mot_metrics.mostly_lost += scene_mot_metrics.mostly_lost
        full_mot_metrics.mostly_tracked += scene_mot_metrics.mostly_tracked
        full_mot_metrics.partly_tracked += scene_mot_metrics.partly_tracked
        full_mot_metrics.true_positives += scene_mot_metrics.true_positives
        full_mot_metrics.false_positives += scene_mot_metrics.false_positives
        full_mot_metrics.false_negatives += scene_mot_metrics.false_negatives

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
    full_mot_metrics.precision = full_mot_metrics.true_positives / (
        full_mot_metrics.true_positives + full_mot_metrics.false_positives
    )
    full_mot_metrics.recall = full_mot_metrics.true_positives / (
        full_mot_metrics.true_positives + full_mot_metrics.false_negatives
    )
    full_mot_metrics.f1 = (2 * full_mot_metrics.precision * full_mot_metrics.recall) / (
        full_mot_metrics.precision + full_mot_metrics.recall
    )
    full_mot_metrics_fname = dst_dir / "mot_metrics.json"
    with open(full_mot_metrics_fname, "w") as fp:
        json.dump(full_mot_metrics.__dict__, fp, indent=4)
    # _plt_error_by_dist(df, dst_dir, args.save)
    # histogram of errors w/o outliers
    # errors vs distance
