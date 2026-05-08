"""Aggregate metrics from completed runs into article tables.

Reads:
    runs/<run_name>_val/<csv files>            -- ultralytics val output
    results/<run_name>.json                    -- evaluate_size.py output

Writes:
    results/table2_overall.csv                 -- P, R, F1, mAP@0.5
    results/table3_recall_by_size.csv          -- Recall per size group
    results/localization_quality.csv           -- centre err per size group

Run example:
    python compare_runs.py \\
        --runs yolov8s_baseline yolov8s_adaptive yolo11s_baseline yolo11s_adaptive \\
        --labels "YOLOv8s baseline" "YOLOv8s + adaptive loss" \\
                  "YOLOv11s baseline" "YOLOv11s + adaptive loss"
"""

import argparse
import csv
import json
import os
from pathlib import Path

import yaml
from ultralytics import YOLO


def get_overall_metrics(weights: str, data_yaml: str = "data/data.yaml") -> dict:
    model = YOLO(weights)
    m = model.val(data=data_yaml, verbose=False, plots=False)
    p, r = float(m.box.mp), float(m.box.mr)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "precision": round(p, 4),
        "recall":    round(r, 4),
        "f1":        round(f1, 4),
        "map50":     round(float(m.box.map50), 4),
        "map50_95":  round(float(m.box.map), 4),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs",   nargs="+", required=True,
                   help="run names (folder names under runs/)")
    p.add_argument("--labels", nargs="+", default=None,
                   help="display labels (same order as --runs)")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--runs-dir",    default="runs")
    args = p.parse_args()

    labels = args.labels or args.runs
    assert len(labels) == len(args.runs), "labels count must match runs count"

    os.makedirs(args.results_dir, exist_ok=True)

    # ---- Table 2: overall metrics ----
    t2 = []
    for run, label in zip(args.runs, labels):
        weights = Path(args.runs_dir) / run / "weights" / "best.pt"
        if not weights.exists():
            print(f"[WARN] missing {weights}, skip")
            continue
        print(f"[INFO] {label}: validating {weights}")
        metrics = get_overall_metrics(str(weights))
        t2.append({"model": label, **metrics})

    t2_path = Path(args.results_dir) / "table2_overall.csv"
    with open(t2_path, "w", newline="") as f:
        if t2:
            w = csv.DictWriter(f, fieldnames=list(t2[0].keys()))
            w.writeheader(); w.writerows(t2)
    print(f"[INFO] saved {t2_path}")

    # ---- Table 3: recall by size + localization quality ----
    t3 = []
    loc = []
    for run, label in zip(args.runs, labels):
        size_json = Path(args.results_dir) / f"{run}.json"
        if not size_json.exists():
            print(f"[WARN] missing {size_json} (run evaluate_size.py first), skip")
            continue
        with open(size_json) as f:
            data = json.load(f)
        t3.append({
            "model":      label,
            "very_small": data["very_small"]["recall"],
            "small":      data["small"]["recall"],
            "medium":     data["medium"]["recall"],
            "large":      data["large"]["recall"],
        })
        loc.append({
            "model":      label,
            "very_small": data["very_small"]["centre_err_mean"],
            "small":      data["small"]["centre_err_mean"],
            "medium":     data["medium"]["centre_err_mean"],
        })

    t3_path = Path(args.results_dir) / "table3_recall_by_size.csv"
    with open(t3_path, "w", newline="") as f:
        if t3:
            w = csv.DictWriter(f, fieldnames=list(t3[0].keys()))
            w.writeheader(); w.writerows(t3)
    print(f"[INFO] saved {t3_path}")

    loc_path = Path(args.results_dir) / "localization_quality.csv"
    with open(loc_path, "w", newline="") as f:
        if loc:
            w = csv.DictWriter(f, fieldnames=list(loc[0].keys()))
            w.writeheader(); w.writerows(loc)
    print(f"[INFO] saved {loc_path}")

    # ---- print summary ----
    if t2:
        print("\n=== Table 2: Overall comparison ===")
        print(f"{'model':<32} {'P':>6} {'R':>6} {'F1':>6} {'mAP@.5':>7}")
        for r in t2:
            print(f"{r['model']:<32} {r['precision']:>6} {r['recall']:>6} "
                  f"{r['f1']:>6} {r['map50']:>7}")

    if t3:
        print("\n=== Table 3: Recall by size group ===")
        print(f"{'model':<32} {'<=16':>7} {'16-32':>7} {'32-64':>7}")
        for r in t3:
            print(f"{r['model']:<32} {r['very_small']:>7} "
                  f"{r['small']:>7} {r['medium']:>7}")


if __name__ == "__main__":
    main()
