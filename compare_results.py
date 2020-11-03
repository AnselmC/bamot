import argparse
from pathlib import Path
from pprint import pprint

import yaml

MAIN_METRICS = {
    "median": {
        "total": ["total", "median"],
        "car": ["obj-type", "car", "median"],
        "ped": ["obj-type", "pedestrian", "median"],
        "reverse": False,
    },
    "median-of-object-mean": {
        "total": ["per-obj", "all", "median-of-mean"],
        "car": ["obj-type", "car", "all", "median-of-mean"],
        "ped": ["obj-type", "pedestrian", "all", "median-of-mean"],
        "reverse": False,
    },
    "outlier-ratio": {
        "total": ["per-obj", "outlier-ratio"],
        "car": ["obj-type", "car", "outlier-ratio"],
        "ped": ["obj-type", "pedestrian", "outlier-ratio"],
        "reverse": False,
    },
}


def _get_nested(d, keys):
    for key in keys:
        d = d[key]
    return d


def _get_n_best(reports, keys, n, reverse=False):
    flattened = {
        fname: _get_nested(summary, keys) for fname, summary in reports.items()
    }
    best = {
        i + 1: (k, v)
        for i, (k, v) in enumerate(
            sorted(flattened.items(), key=lambda x: x[1], reverse=reverse)[:n]
        )
    }
    return best


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

    n = args.n

    for metric_name, info in MAIN_METRICS.items():
        print("*" * 30)
        print(f"Metric: {metric_name.replace('-', ' ').capitalize()}")
        reverse_order = info.pop("reverse")
        for obj_type, keys in info.items():
            print("-" * 30)
            print(obj_type.capitalize())
            best = _get_n_best(reports, keys, n, reverse=reverse_order)
            print(f"{n} best:")
            pprint(best, indent=4)
