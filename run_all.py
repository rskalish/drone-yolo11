"""Run the full experiment pipeline in one command.

Usage:
    python run_all.py               # all 4 runs, 80 epochs
    python run_all.py --epochs 30   # quick test
    python run_all.py --batch 8     # less VRAM
"""

import argparse
import os
import subprocess
import sys


RUNS = [
    {"model": "yolov8s.pt",  "name": "yolov8s_baseline", "adaptive": False},
    {"model": "yolov8s.pt",  "name": "yolov8s_adaptive",  "adaptive": True},
    {"model": "yolo11s.pt",  "name": "yolo11s_baseline",  "adaptive": False},
    {"model": "yolo11s.pt",  "name": "yolo11s_adaptive",  "adaptive": True},
]

LABELS = [
    "YOLOv8s baseline",
    "YOLOv8s + adaptive loss",
    "YOLOv11s baseline",
    "YOLOv11s + adaptive loss",
]


def run(cmd: list[str]):
    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",   type=int, default=80)
    p.add_argument("--batch",    type=int, default=16)
    p.add_argument("--skip-train",   action="store_true", help="skip training, only evaluate")
    p.add_argument("--skip-figures", action="store_true", help="skip figure generation")
    args = p.parse_args()

    py = sys.executable
    os.makedirs("results", exist_ok=True)

    # ── 1. Training ──────────────────────────────────────────────────────────
    if not args.skip_train:
        print("\n" + "=" * 60)
        print("STEP 1 / 4 — Training")
        print("=" * 60)
        for cfg in RUNS:
            cmd = [
                py, "train.py",
                "--model",   cfg["model"],
                "--name",    cfg["name"],
                "--epochs",  str(args.epochs),
                "--batch",   str(args.batch),
            ]
            if cfg["adaptive"]:
                cmd.append("--adaptive")
            run(cmd)

    # ── 2. Dataset stats (Table 1) ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 / 4 — Dataset statistics (Table 1)")
    print("=" * 60)
    run([py, "dataset_stats.py"])

    # ── 3. Per-size evaluation ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 / 4 — Per-size Recall evaluation")
    print("=" * 60)
    for cfg in RUNS:
        weights = os.path.join("runs", cfg["name"], "weights", "best.pt")
        if not os.path.exists(weights):
            print(f"[WARN] {weights} not found, skipping")
            continue
        run([
            py, "evaluate_size.py",
            "--weights", weights,
            "--out",     f"results/{cfg['name']}.json",
        ])

    # ── 4. Comparison tables ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 / 4 — Compare runs + generate figures")
    print("=" * 60)
    run_names = [cfg["name"] for cfg in RUNS]
    run([
        py, "compare_runs.py",
        "--runs",   *run_names,
        "--labels", *LABELS,
    ])

    # ── 5. Figures ────────────────────────────────────────────────────────────
    if not args.skip_figures:
        run([
            py, "figures.py",
            "--runs",   *run_names,
            "--labels", *[l.replace(" + ", " +\n") for l in LABELS],
            "--n-examples", "6",
        ])

    print("\n" + "=" * 60)
    print("ALL DONE")
    print("Tables : results/table1_distribution.csv")
    print("         results/table2_overall.csv")
    print("         results/table3_recall_by_size.csv")
    print("Figures: figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
