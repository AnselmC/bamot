import argparse
from pathlib import Path
from pprint import pprint

import numpy as np
import yaml

MAIN_METRICS = {
    "offline-median": {
        "total": ["total", "offline", "median"],
        "car": ["obj-type", "car", "offline", "median"],
        "ped": ["obj-type", "pedestrian", "offline", "median"],
        "reverse": False,
    },
    "offline-median-of-object-mean": {
        "total": ["per-obj", "all", "offline", "median-of-mean"],
        "car": ["obj-type", "car", "all", "offline", "median-of-mean"],
        "ped": ["obj-type", "pedestrian", "all", "offline", "median-of-mean"],
        "reverse": False,
    },
    "offline-outlier-ratio": {
        "total": ["per-obj", "outlier-ratio", "offline"],
        "car": ["obj-type", "car", "outlier-ratio", "offline"],
        "ped": ["obj-type", "pedestrian", "outlier-ratio", "offline"],
        "reverse": False,
    },
    "online-median": {
        "total": ["total", "online", "median"],
        "car": ["obj-type", "car", "online", "median"],
        "ped": ["obj-type", "pedestrian", "online", "median"],
        "reverse": False,
    },
    "online-median-of-object-mean": {
        "total": ["per-obj", "all", "online", "median-of-mean"],
        "car": ["obj-type", "car", "all", "online", "median-of-mean"],
        "ped": ["obj-type", "pedestrian", "all", "online", "median-of-mean"],
        "reverse": False,
    },
    "online-outlier-ratio": {
        "total": ["per-obj", "outlier-ratio", "online"],
        "car": ["obj-type", "car", "outlier-ratio", "online"],
        "ped": ["obj-type", "pedestrian", "outlier-ratio", "online"],
        "reverse": False,
    },
    "tracked-ratio": {
        "total": ["total", "tracked_ratio"],
        "car": ["obj-type", "car", "tracked_ratio"],
        "ped": ["obj-type", "pedestrian", "tracked_ratio"],
        "reverse": True,
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
    sorted_flattened = sorted(flattened.items(), key=lambda x: x[1], reverse=reverse)[
        :n
    ]
    index = 0
    current_val = -1
    order = []
    for t in sorted_flattened:
        if t[1] != current_val:
            current_val = t[1]
            index += 1
        order.append(index)

    best = [(i, (k, v)) for i, (k, v) in zip(order, sorted_flattened)]
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
    overall_best = {}

    for metric_name, info in MAIN_METRICS.items():
        print("*" * 30)
        print(f"Metric: {metric_name.replace('-', ' ').capitalize()}")
        reverse_order = info.pop("reverse")
        for obj_type, keys in info.items():
            if overall_best.get(obj_type) is None:
                overall_best[obj_type] = {}
            print("-" * 30)
            print(obj_type.capitalize())
            best = _get_n_best(reports, keys, n, reverse=reverse_order)
            for place, (name, _) in best:
                if overall_best[obj_type].get(name) is None:
                    overall_best[obj_type][name] = []
                overall_best[obj_type][name].append(place)
            print(f"{n} best:")
            pprint(best, indent=4)
    for obj_type, d in overall_best.items():
        names_mean = []
        for name, scores in d.items():
            names_mean.append((name, np.mean(scores)))
        overall_best[obj_type] = sorted(names_mean, key=lambda x: x[1])
    print("OVERALL BEST")
    pprint(overall_best)
