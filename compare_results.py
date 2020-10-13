import argparse
from pathlib import Path

import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "results",
        help="The evaluation result directories to compare (need to have a `full_eval` directory which contains a `result.yaml` file)",
        nargs="*",
    )
    args = parser.parse_args()
    if len(args.results) < 2:
        print("Need at least two directories/files to compare")
        exit()

    for r in args.results:
        fname = Path(r) / "full_eval" / "report.yaml"
        reports = {}
        if not fname.exists():
            print(f"{fname} does not exist...")
            exit()
            with open(fname.as_posix(), "r") as fp:
                rep = yaml.load(fp, Loader=yaml.FullLoader)
            reports[fname] = rep["summary"]

    # get fname
    # filter by
    medians = {fname: summary["total"]["median"] for fname, summary in reports.items()}
    best_median = [(k, v) for k, v in sorted(medians.items(), key=lambda x: x[1])][0]
    print(f"BEST MEDIAN: {best_median}")
    means = {fname: summary["total"]["means"] for fname, summary in reports.items()}
    best_mean = [(k, v) for k, v in sorted(means.items(), key=lambda x: x[1])][0]
    print(f"BEST MEAN: {best_mean}")

    medians_car = {
        fname: summary["obj-type"]["car"]["median"]
        for fname, summary in reports.items()
    }
    best_median_car = [
        (k, v) for k, v in sorted(medians_car.items(), key=lambda x: x[1])
    ][0]
    print(f"BEST MEDIAN CAR: {best_median_car}")
    means_car = {
        fname: summary["obj-type"]["car"]["means"] for fname, summary in reports.items()
    }
    best_mean_car = [(k, v) for k, v in sorted(means_car.items(), key=lambda x: x[1])][
        0
    ]
    print(f"BEST MEAN CAR: {best_mean_car}")

    medians_pedestrian = {
        fname: summary["obj-type"]["pedestrian"]["median"]
        for fname, summary in reports.items()
    }
    best_median_pedestrian = [
        (k, v) for k, v in sorted(medians_pedestrian.items(), key=lambda x: x[1])
    ][0]
    print(f"BEST MEDIAN PEDESTRIAN: {best_median_pedestrian}")
    means_pedestrian = {
        fname: summary["obj-type"]["pedestrian"]["means"]
        for fname, summary in reports.items()
    }
    best_mean_pedestrian = [
        (k, v) for k, v in sorted(means_pedestrian.items(), key=lambda x: x[1])
    ][0]
    print(f"BEST MEAN PEDESTRIAN: {best_mean_pedestrian}")
