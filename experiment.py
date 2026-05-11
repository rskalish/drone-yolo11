"""Run the full experiment end-to-end: train -> evaluate -> tables -> figures.

This is the single entry point for reproducing the article results.

Local (PyCharm CE / M3 Air):  ~10 epochs per model
    python experiment.py

Colab (T4 / L4 / A100):       80 epochs per model
    python experiment.py            # epochs auto-detected
    python experiment.py --epochs 50

Skip already-done steps:
    python experiment.py --skip-train          # only evaluate + figures
    python experiment.py --skip-figures
"""

import argparse
import subprocess
import sys
from pathlib import Path

from config import RUNS, env_summary, get_results_dir, get_runs_dir


def run(cmd: list[str]):
    print(f"\n>>> {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None,
                   help="override epochs (default: 80 in Colab, 10 local)")
    p.add_argument("--batch",  type=int, default=None,
                   help="override batch (default: 16 in Colab, 8 local)")
    p.add_argument("--skip-train",   action="store_true")
    p.add_argument("--skip-figures", action="store_true")
    args = p.parse_args()

    print(env_summary())
    py = sys.executable
    runs_dir    = get_runs_dir()
    results_dir = get_results_dir()

    # ── 1. Training ──────────────────────────────────────────────────────────
    if not args.skip_train:
        print("\n" + "=" * 60)
        print("STEP 1 / 4 — Training all 4 models")
        print("=" * 60)
        cmd = [py, "train.py", "--all"]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        if args.batch is not None:
            cmd += ["--batch", str(args.batch)]
        run(cmd)

    # ── 2. Dataset statistics (Table 1) ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 / 4 — Dataset statistics (Table 1)")
    print("=" * 60)
    run([py, "dataset_stats.py"])

    # ── 3. Per-size evaluation ────────────────────────────────────────────────
    # Each run is evaluated on its own dataset (v8 dataset for YOLOv8 models,
    # v11 dataset for YOLOv11 models).
    print("\n" + "=" * 60)
    print("STEP 3 / 4 — Per-size Recall evaluation")
    print("=" * 60)
    for cfg in RUNS:
        weights = runs_dir / cfg["name"] / "weights" / "best.pt"
        if not weights.exists():
            print(f"[WARN] {weights} missing, skipping")
            continue
        run([
            py, "evaluate_size.py",
            "--weights", str(weights),
            "--out",     str(results_dir / f"{cfg['name']}.json"),
        ])

    # ── 4. Tables and figures ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 / 4 — Tables and figures")
    print("=" * 60)
    run([py, "compare_runs.py"])
    if not args.skip_figures:
        run([py, "figures.py"])

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Tables : {results_dir}")
    print(f"  Figures: {Path('figures' if str(results_dir).startswith('results') else results_dir.parent / 'figures')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
