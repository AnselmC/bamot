import argparse
from pathlib import Path
from pprint import pprint

import yaml


def _get_nested(d, keys):
    for key in keys:
        d = d[key]
    return d


def _get_n_best(reports, keys, n):
    medianss = {
        fname: _get_nested(summary, keys)["median"]
        for fname, summary in reports.items()
    }
    best_medianss = [(k, v) for k, v in sorted(medianss.items(), key=lambda x: x[1])][
        :n
    ]
    meanss = {
        fname: _get_nested(summary, keys)["mean"] for fname, summary in reports.items()
    }
    best_meanss = [(k, v) for k, v in sorted(meanss.items(), key=lambda x: x[1])][:n]

    return best_medianss, best_meanss


def get_metrics(summary):
    print("-" * 30)
    print("Median (total)")
    print(f'Total: {summary["total"]["median"]:.2f}')
    print(f'Cars: {summary["obj-type"]["car"]["median"]:.2f}')
    print(f'Peds: {summary["obj-type"]["pedestrian"]["median"]:.2f}')
    print("Median (obj means)")
    print(f"Total: {summary['per-obj']['all']['median-of-mean']:.2f}")
    print(f"Cars: {summary['obj-type']['car']['all']['median-of-mean']:.2f}")
    print(
        f"Pedestrian: {summary['obj-type']['pedestrian']['all']['median-of-mean']:.2f}"
    )
    print("Outlier ratio")
    print(f"Total: {summary['per-obj']['outlier-ratio']:.2f}")
    print(f"Cars: {summary['obj-type']['car']['outlier-ratio']:.2f}")
    print(f"Pedestrian: {summary['obj-type']['pedestrian']['outlier-ratio']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "results",
        help="The evaluation result directories to compare (need to have a `full_eval` directory which contains a `result.yaml` file)",
        nargs="*",
    )
    parser.add_argument(
        "-n", help="Return n best results (defaults to 1)", default=1, type=int
    )
    args = parser.parse_args()

    reports = {}
    for r in args.results:
        fname = Path(r) / "full_eval" / "report.yaml"
        if not fname.exists():
            print(f"{fname} does not exist...")
            continue
        with open(fname.as_posix(), "r") as fp:
            rep = yaml.load(fp, Loader=yaml.FullLoader)
        reports[r.split("/")[-1]] = rep["summary"]

    if len(reports) < 2:
        print("Need at least two directories/files to compare")
        exit()

    # get fname
    # filter by
    for fname, summary in reports.items():
        print("+" * 30)
        print(fname)
        get_metrics(summary)
    n = args.n
    best_medians, best_means = _get_n_best(reports, ["total"], n)
    print(f"BEST MEDIANS: {best_medians}")
    print(f"BEST MEANS: {best_means}")

    best_medians_car, best_means_car = _get_n_best(reports, ["obj-type", "car"], n)
    print(f"BEST MEDIANS CAR: {best_medians_car}")
    print(f"BEST MEANS CAR: {best_means_car}")

    best_medians_ped, best_means_ped = _get_n_best(
        reports, ["obj-type", "pedestrian"], n
    )
    print(f"BEST MEDIANS PEDESTRIAN: {best_medians_ped}")
    print(f"BEST MEANS PEDESTRIAN: {best_means_ped}")

    best_medians_close, best_means_close = _get_n_best(
        reports, ["distance", "close"], n
    )
    print(f"BEST MEDIANS CLOSE: {best_medians_close}")
    print(f"BEST MEANS CLOSE: {best_means_close}")

    best_medians_mid, best_means_mid = _get_n_best(reports, ["distance", "mid"], n)
    print(f"BEST MEDIANS MID: {best_medians_mid}")
    print(f"BEST MEANS MID: {best_means_mid}")

    best_medians_far, best_means_far = _get_n_best(reports, ["distance", "far"], n)
    print(f"BEST MEDIANS FAR: {best_medians_far}")
    print(f"BEST MEANS FAR: {best_means_far}")
