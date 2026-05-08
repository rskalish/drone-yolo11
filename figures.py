"""Generate publication-ready figures for the article.

Produces:
    figures/fig_recall_by_size.png   -- bar chart, Table 3 visualization
    figures/fig_overall_metrics.png  -- bar chart, Table 2 visualization
    figures/fig_training_curves.png  -- merged loss + mAP curves
    figures/fig_pr_curves.png        -- copies of ultralytics PR curves
    figures/detections/<run>/<img>   -- test images with predicted boxes

Run after compare_runs.py and after each run has its results JSON.
"""

import argparse
import csv
import glob
import os
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

plt.rcParams.update({
    "font.size": 11, "figure.dpi": 130, "savefig.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False,
})


def fig_recall_by_size(results_dir: str, fig_dir: str):
    csv_path = Path(results_dir) / "table3_recall_by_size.csv"
    if not csv_path.exists():
        print(f"[WARN] {csv_path} missing"); return
    df = pd.read_csv(csv_path)
    groups = ["very_small", "small", "medium"]
    labels = ["≤16×16", "16–32", "32–64"]
    x = np.arange(len(groups))
    w = 0.8 / max(len(df), 1)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, row in df.iterrows():
        is_adapt = "adaptive" in row["model"].lower()
        color = "#d6604d" if is_adapt else "#4393c3"
        bars = ax.bar(x + i * w - 0.4 + w / 2, [row[g] for g in groups],
                      w, label=row["model"], color=color,
                      edgecolor="black", linewidth=0.4,
                      hatch="//" if is_adapt else None)
        for b, g in zip(bars, groups):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"{row[g]:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Recall"); ax.set_xlabel("Size group, px")
    ax.set_title("Recall by object size — baseline vs adaptive loss")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    out = Path(fig_dir) / "fig_recall_by_size.png"
    plt.tight_layout(); plt.savefig(out); plt.close()
    print(f"[INFO] saved {out}")


def fig_overall_metrics(results_dir: str, fig_dir: str):
    csv_path = Path(results_dir) / "table2_overall.csv"
    if not csv_path.exists():
        print(f"[WARN] {csv_path} missing"); return
    df = pd.read_csv(csv_path)
    metrics = ["precision", "recall", "f1", "map50"]
    labels = ["Precision", "Recall", "F1", "mAP@0.5"]
    x = np.arange(len(metrics))
    w = 0.8 / max(len(df), 1)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, row in df.iterrows():
        is_adapt = "adaptive" in row["model"].lower()
        color = "#d6604d" if is_adapt else "#4393c3"
        bars = ax.bar(x + i * w - 0.4 + w / 2, [row[m] for m in metrics],
                      w, label=row["model"], color=color,
                      edgecolor="black", linewidth=0.4,
                      hatch="//" if is_adapt else None)
        for b, m in zip(bars, metrics):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"{row[m]:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("Value")
    ax.set_title("Overall detection metrics — baseline vs adaptive loss")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    out = Path(fig_dir) / "fig_overall_metrics.png"
    plt.tight_layout(); plt.savefig(out); plt.close()
    print(f"[INFO] saved {out}")


def fig_training_curves(runs: list[str], labels: list[str],
                        runs_dir: str, fig_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for run, label in zip(runs, labels):
        csv_path = Path(runs_dir) / run / "results.csv"
        if not csv_path.exists():
            print(f"[WARN] {csv_path} missing"); continue
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        ep = df["epoch"]
        is_adapt = "adaptive" in label.lower()
        ls = "--" if is_adapt else "-"
        # box loss column may be "train/box_loss"
        box_col = next((c for c in df.columns if "box_loss" in c and "train" in c), None)
        map_col = next((c for c in df.columns if "mAP50(B)" in c or "mAP50" == c), None)
        if box_col:
            axes[0].plot(ep, df[box_col], ls, label=label)
        if map_col:
            axes[1].plot(ep, df[map_col], ls, label=label)

    axes[0].set_title("Training box loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].grid(linestyle=":", alpha=0.4); axes[0].legend(fontsize=9)
    axes[1].set_title("Validation mAP@0.5")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("mAP@0.5")
    axes[1].grid(linestyle=":", alpha=0.4); axes[1].legend(fontsize=9)

    out = Path(fig_dir) / "fig_training_curves.png"
    plt.tight_layout(); plt.savefig(out); plt.close()
    print(f"[INFO] saved {out}")


def copy_pr_curves(runs: list[str], runs_dir: str, fig_dir: str):
    out_dir = Path(fig_dir) / "pr_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        src = Path(runs_dir) / run / "PR_curve.png"
        if src.exists():
            dst = out_dir / f"{run}_PR_curve.png"
            shutil.copy(src, dst)
            print(f"[INFO] copied {dst}")


def render_detections(runs: list[str], labels: list[str],
                      runs_dir: str, fig_dir: str,
                      n_examples: int = 6, conf: float = 0.25):
    """Render side-by-side detection comparisons on test images."""
    test_imgs = sorted(glob.glob("data/test/images/*.jpg") +
                       glob.glob("data/test/images/*.png"))[:n_examples]
    if not test_imgs:
        print("[WARN] no test images"); return

    out_root = Path(fig_dir) / "detections"
    out_root.mkdir(parents=True, exist_ok=True)

    for run, label in zip(runs, labels):
        weights = Path(runs_dir) / run / "weights" / "best.pt"
        if not weights.exists():
            print(f"[WARN] missing {weights}"); continue
        model = YOLO(str(weights))
        run_out = out_root / run
        run_out.mkdir(exist_ok=True)
        for img_path in test_imgs:
            r = model.predict(img_path, conf=conf, verbose=False)[0]
            annotated = r.plot()  # BGR numpy
            cv2.imwrite(str(run_out / Path(img_path).name), annotated)
        print(f"[INFO] saved detections to {run_out}")

    # composite panel: rows = images, cols = runs
    n_runs = len(runs)
    fig, axes = plt.subplots(n_examples, n_runs,
                              figsize=(3.6 * n_runs, 3.6 * n_examples))
    if n_examples == 1:
        axes = np.array([axes])
    if n_runs == 1:
        axes = axes.reshape(-1, 1)
    for i, img_path in enumerate(test_imgs):
        for j, (run, label) in enumerate(zip(runs, labels)):
            p = out_root / run / Path(img_path).name
            if not p.exists():
                continue
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            ax = axes[i, j]
            ax.imshow(img); ax.axis("off")
            if i == 0:
                ax.set_title(label, fontsize=10)

    out = Path(fig_dir) / "fig_detections_panel.png"
    plt.tight_layout(); plt.savefig(out); plt.close()
    print(f"[INFO] saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs",   nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--runs-dir",    default="runs")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--fig-dir",     default="figures")
    p.add_argument("--n-examples",  type=int, default=6)
    args = p.parse_args()

    labels = args.labels or args.runs
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    fig_recall_by_size(args.results_dir, args.fig_dir)
    fig_overall_metrics(args.results_dir, args.fig_dir)
    fig_training_curves(args.runs, labels, args.runs_dir, args.fig_dir)
    copy_pr_curves(args.runs, args.runs_dir, args.fig_dir)
    render_detections(args.runs, labels, args.runs_dir, args.fig_dir,
                      n_examples=args.n_examples)


if __name__ == "__main__":
    main()
