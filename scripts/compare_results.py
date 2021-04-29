import argparse
import json
from collections import defaultdict
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

MOTS_METRICS = {
    "false_positives": {"reverse": False},
    "true_positives": {"reverse": True},
    "false_negatives": {"reverse": False},
    "mostly_tracked": {"reverse": True},
    "mostly_lost": {"reverse": False},
    "partly_tracked": {"reverse": True},
    "precision": {"reverse": True},
    "recall": {"reverse": True},
    "f1": {"reverse": True},
}


def _get_nested(d, keys):
    for key in keys:
        d = d[key]
    return d


def _flatten_reports(reports, keys):
    return {fname: _get_nested(summary, keys) for fname, summary in reports.items()}


def _get_n_best(d, n, reverse=False):
    sorted_d = sorted(d.items(), key=lambda x: x[1], reverse=reverse)[:n]
    index = 0
    current_val = -1
    order = []
    for t in sorted_d:
        if t[1] != current_val:
            current_val = t[1]
            index += 1
        order.append(index)

    best = [(i, (k, v)) for i, (k, v) in zip(order, sorted_d)]
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
    mots = {}
    for r in args.results:
        report_fname = Path(r) / "full_eval" / "report.yaml"
        if not report_fname.exists():
            print(f"{report_fname} does not exist...")
            continue
        with open(report_fname, "r") as fp:
            rep = yaml.load(fp, Loader=yaml.FullLoader)
        reports[r.split("/")[-1]] = rep["summary"]

        mots_fname = Path(r) / "full_eval" / "mot_metrics.json"
        if not mots_fname.exists():
            print(f"{mots_fname} does not exist...")
            continue
        with open(mots_fname, "r") as fp:
            m = json.load(fp)
        mots[r.split("/")[-1]] = m

    if len(reports) < 2:
        print("Need at least two directories/files to compare")
        exit()

    n = args.n
    overall_best = {}

    print("*" * 60)
    print("CUSTOM 3D EVALUATION")

    for metric_name, info in MAIN_METRICS.items():
        print("=" * 30)
        print(f"Metric: {metric_name.replace('-', ' ').capitalize()}")
        reverse_order = info.pop("reverse")
        for obj_type, keys in info.items():
            if overall_best.get(obj_type) is None:
                overall_best[obj_type] = {}
            print("-" * 30)
            print(obj_type.capitalize())
            best = _get_n_best(
                _flatten_reports(reports, keys), n, reverse=reverse_order
            )
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
    print("*" * 60)
    print("CUSTOM MOTS EVALUATION")
    scores = defaultdict(list)
    for metric_name, info in MOTS_METRICS.items():
        print(f"Metric: {metric_name.replace('-', ' ').capitalize()}")
        print("=" * 30)
        print(f"Metric: {metric_name.replace('-', ' ').capitalize()}")
        reverse_order = info.pop("reverse")
        best = _get_n_best(
            {name: d[metric_name] for name, d in mots.items()}, n, reverse=reverse_order
        )
        print(f"{n} best:")
        pprint(best, indent=4)
        for place, (name, _) in best:
            scores[name].append(place)
    overall_best = {}
    for name, scores in scores.items():
        overall_best[name] = np.mean(scores)
    overall_best = sorted(overall_best.items(), key=lambda x: x[1])
    print("OVERALL BEST")
    pprint(overall_best)
