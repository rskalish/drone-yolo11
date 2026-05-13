"""Re-evaluate all 6 trained models at a low confidence threshold (default 0.10).

Motivation
----------
Adaptive loss (target_scores *= w/W_MAX) shifts confidence calibration for
large objects — they ARE detected but with lower confidence (~0.10–0.20),
so the default conf=0.25 threshold filters them out and recall@conf=0.25
reads as ~0.07. mAP@0.5 (threshold-independent) shows the model is fine
(≈0.89 for adaptive vs ≈0.93 for baseline). This script re-runs per-size
evaluation at conf=0.10 so the article can report two complementary tables:

  Table 3a: recall @ conf=0.25 (standard)
  Table 3b: recall @ conf=0.10 (adaptive-calibration-aware)

Usage
-----
    python evaluate_conf_sweep.py              # conf=0.10, all runs in config.RUNS
    python evaluate_conf_sweep.py --conf 0.05  # even lower threshold
    python evaluate_conf_sweep.py --runs yolov8s_adaptive_w2 yolo11s_adaptive_w2

Outputs
-------
    results/<run>_conf010.json
    results/table3_recall_by_size_conf010.csv
    results/localization_quality_conf010.csv
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from config import LABELS, RUNS, get_results_dir, get_runs_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conf",   type=float, default=0.10)
    p.add_argument("--iou",    type=float, default=0.5)
    p.add_argument("--runs",   nargs="+", default=None,
                   help="run names (default: all from config.RUNS)")
    p.add_argument("--labels", nargs="+", default=None)
    args = p.parse_args()

    runs_dir    = get_runs_dir()
    results_dir = get_results_dir()
    conf_tag    = f"conf{int(args.conf * 1000):03d}"   # 0.10 -> conf100, 0.05 -> conf050

    if args.runs:
        cfg_map = {c["name"]: c for c in RUNS}
        labels  = args.labels or args.runs
        triples = [(rn, lb, cfg_map.get(rn)) for rn, lb in zip(args.runs, labels)]
    else:
        triples = [(c["name"], lb, c) for c, lb in zip(RUNS, LABELS)]

    py = sys.executable

    # ── 1. Run evaluate_size.py at the new threshold for each model ──────────
    for run_name, label, _cfg in triples:
        weights = runs_dir / run_name / "weights" / "best.pt"
        if not weights.exists():
            print(f"[WARN] {weights} missing, skipping")
            continue

        out_json = results_dir / f"{run_name}_{conf_tag}.json"
        print(f"\n[INFO] {label}  conf={args.conf}  → {out_json.name}")
        subprocess.run([
            py, "evaluate_size.py",
            "--weights", str(weights),
            "--conf",    str(args.conf),
            "--iou-thr", str(args.iou),
            "--out",     str(out_json),
        ], check=True)

    # ── 2. Build comparison CSV ───────────────────────────────────────────────
    t3, loc = [], []
    for run_name, label, _ in triples:
        size_json = results_dir / f"{run_name}_{conf_tag}.json"
        if not size_json.exists():
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

    t3_path = results_dir / f"table3_recall_by_size_{conf_tag}.csv"
    if t3:
        with open(t3_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(t3[0].keys()))
            w.writeheader(); w.writerows(t3)
    print(f"\n[INFO] saved {t3_path}")

    loc_path = results_dir / f"localization_quality_{conf_tag}.csv"
    if loc:
        with open(loc_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(loc[0].keys()))
            w.writeheader(); w.writerows(loc)
    print(f"[INFO] saved {loc_path}")

    # ── 3. Console summary ────────────────────────────────────────────────────
    if t3:
        print(f"\n=== Recall by size group  (conf={args.conf}) ===")
        print(f"{'model':<35} {'<=16':>8} {'16-32':>8} {'32-64':>8} {'>64':>8}")
        for r in t3:
            print(f"{r['model']:<35} {r['very_small']:>8} "
                  f"{r['small']:>8} {r['medium']:>8} {r['large']:>8}")


if __name__ == "__main__":
    main()
