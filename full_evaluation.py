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
    dist_summary = {}
    df_close = df.loc[df.distance < 5]
    df_mid = df.loc[(df.distance >= 5) & (df.distance < 30)]
    df_far = df.loc[df.distance >= 30]
    dist_summary["close"] = _get_metrics(df_close)
    dist_summary["mid"] = _get_metrics(df_mid)
    dist_summary["far"] = _get_metrics(df_far)
    summary["distance"] = dist_summary
    summary["obj-lvl"] = _summarize_obj_lvl(df)
    return summary


def _summarize_obj_lvl(df):
    obj_df = df.groupby(["scene", "object_id"])
    num_objs = len(obj_df)
    mean_errors = obj_df.error.mean()
    median_errors = obj_df.error.median()
    obj_summary = {}
    obj_summary["mean-error-lt-5"] = float(100 * (mean_errors < 5).sum() / num_objs)
    obj_summary["mean-error-lt-3"] = float(100 * (mean_errors < 3).sum() / num_objs)
    obj_summary["mean-error-lt-1"] = float(100 * (mean_errors < 1).sum() / num_objs)
    obj_summary["median-error-lt-5"] = float(100 * (median_errors < 5).sum() / num_objs)
    obj_summary["median-error-lt-3"] = float(100 * (median_errors < 3).sum() / num_objs)
    obj_summary["median-error-lt-1"] = float(100 * (median_errors < 1).sum() / num_objs)
    best_objs = {}
    best_objs_mean = []
    for obj in mean_errors.nsmallest(n=10).items():
        best_objs_mean.append(
            {"scene": obj[0][0], "object_id": obj[0][1], "error": obj[1]}
        )
    best_objs_median = []
    for obj in median_errors.nsmallest(n=10).items():
        best_objs_median.append(
            {"scene": obj[0][0], "object_id": obj[0][1], "error": obj[1]}
        )

    worst_objs = {}
    worst_objs_mean = []
    for obj in mean_errors.nlargest(n=10).items():
        worst_objs_mean.append(
            {"scene": obj[0][0], "object_id": obj[0][1], "error": obj[1]}
        )
    worst_objs_median = []
    for obj in median_errors.nlargest(n=10).items():
        worst_objs_median.append(
            {"scene": obj[0][0], "object_id": obj[0][1], "error": obj[1]}
        )

    best_objs["mean"] = best_objs_mean
    best_objs["median"] = best_objs_median
    worst_objs["mean"] = worst_objs_mean
    worst_objs["median"] = worst_objs_median
    obj_summary["best"] = best_objs
    obj_summary["worst"] = worst_objs
    return obj_summary


def _get_metrics(df):
    metrics = {}
    metrics["mean"] = float(df.error.mean())
    metrics["median"] = float(df.error.median())
    metrics["stddev"] = float(df.error.std())
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
    for scene in sorted(df.scene.unique()):
        print(f"Creating summary for scene {scene}")
        scene_df = df[df.scene == scene]
        per_scene_report[scene] = _summarize_df(scene_df)
        print("Created!")
    report["per-scene"] = per_scene_report
    if save:
        import pdb

        pdb.set_trace()
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
    _plt_error_by_dist(df, dst_dir, args.save)
    # histogram of errors w/o outliers
    # errors vs distance
