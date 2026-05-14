"""Generate publication-ready figures for the article.

Outputs into figures/ (or Drive/figures/ in Colab):
    fig_recall_by_size.png      -- bar chart of Table 3
    fig_overall_metrics.png     -- bar chart of Table 2
    fig_training_curves.png     -- merged loss + mAP curves
    fig_detections_panel.png    -- side-by-side detection comparison
    pr_curves/                  -- copies of ultralytics PR curves
    detections/<run>/           -- per-image annotated detections
"""

import argparse
import glob
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

from config import DATA_DIR, LABELS, RUNS, get_figures_dir, get_results_dir, get_runs_dir

plt.rcParams.update({
    "font.size": 11, "figure.dpi": 130, "savefig.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _bar_color(model_name: str) -> tuple:
    is_adapt = "adaptive" in model_name.lower()
    return ("#d6604d" if is_adapt else "#4393c3", "//" if is_adapt else None)


def fig_recall_by_size(results_dir: Path, fig_dir: Path):
    csv_path = results_dir / "table3_recall_by_size.csv"
    if not csv_path.exists():
        print(f"[WARN] {csv_path} missing"); return
    df = pd.read_csv(csv_path)
    groups = ["very_small", "small", "medium"]
    labels = ["≤16×16", "16–32", "32–64"]
    x = np.arange(len(groups))
    w = 0.8 / max(len(df), 1)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, row in df.iterrows():
        color, hatch = _bar_color(row["model"])
        bars = ax.bar(x + i * w - 0.4 + w / 2,
                      [row[g] for g in groups], w,
                      label=row["model"], color=color,
                      edgecolor="black", linewidth=0.4, hatch=hatch)
        for b, g in zip(bars, groups):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"{row[g]:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Recall"); ax.set_xlabel("Size group, px")
    ax.set_title("Recall by object size — baseline vs adaptive loss")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    out = fig_dir / "fig_recall_by_size.png"
    plt.tight_layout(); plt.savefig(out); plt.close()
    print(f"[INFO] saved {out}")


def fig_overall_metrics(results_dir: Path, fig_dir: Path):
    csv_path = results_dir / "table2_overall.csv"
    if not csv_path.exists():
        print(f"[WARN] {csv_path} missing"); return
    df = pd.read_csv(csv_path)
    metrics = ["precision", "recall", "f1", "map50", "map50_95"]
    labels = ["Precision", "Recall", "F1", "mAP@0.5", "mAP@0.5:0.95"]
    x = np.arange(len(metrics))
    w = 0.8 / max(len(df), 1)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, row in df.iterrows():
        color, hatch = _bar_color(row["model"])
        bars = ax.bar(x + i * w - 0.4 + w / 2,
                      [row[m] for m in metrics], w,
                      label=row["model"], color=color,
                      edgecolor="black", linewidth=0.4, hatch=hatch)
        for b, m in zip(bars, metrics):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"{row[m]:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0); ax.set_ylabel("Value")
    ax.set_title("Overall detection metrics — baseline vs adaptive loss")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    out = fig_dir / "fig_overall_metrics.png"
    plt.tight_layout(); plt.savefig(out); plt.close()
    print(f"[INFO] saved {out}")


def fig_training_curves(runs, labels, runs_dir: Path, fig_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for run, label in zip(runs, labels):
        csv_path = runs_dir / run / "results.csv"
        if not csv_path.exists():
            print(f"[WARN] {csv_path} missing"); continue
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        ep = df["epoch"]
        ls = "--" if "adaptive" in label.lower() else "-"
        box_col = next((c for c in df.columns if "box_loss" in c and "train" in c), None)
        map_col = next((c for c in df.columns if "mAP50(B)" in c or c == "mAP50"), None)
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

    out = fig_dir / "fig_training_curves.png"
    plt.tight_layout(); plt.savefig(out); plt.close()
    print(f"[INFO] saved {out}")


def copy_pr_curves(runs, runs_dir: Path, fig_dir: Path):
    out_dir = fig_dir / "pr_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        src = runs_dir / run / "PR_curve.png"
        if src.exists():
            shutil.copy(src, out_dir / f"{run}_PR_curve.png")
            print(f"[INFO] copied {out_dir / f'{run}_PR_curve.png'}")


def _pick_drone_examples(n: int = 6):
    """Pick test images that ACTUALLY contain drone annotations.

    The test set includes background images (rooms, planes, birds) used as
    hard negatives during training — those should not appear in the article
    detection panel. We filter by non-empty label files, then bucket by box
    size and sample evenly across size groups so the panel covers the full
    scale spectrum (very_small / small / medium / large).
    """
    img_dir = DATA_DIR / "test" / "images"
    lbl_dir = DATA_DIR / "test" / "labels"
    imgs = sorted(glob.glob(str(img_dir / "*.jpg")) +
                  glob.glob(str(img_dir / "*.png")))

    buckets = {"very_small": [], "small": [], "medium": [], "large": []}
    for img_path in imgs:
        lbl = lbl_dir / (Path(img_path).stem + ".txt")
        if not lbl.exists():
            continue
        with open(lbl) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            continue                      # background image — skip
        # Use the largest box in the image to pick a representative bucket
        max_area = 0.0
        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                continue
            _, _, _, w, h = parts[:5]      # YOLO normalized format
            max_area = max(max_area, float(w) * float(h))
        # buckets by area fraction of the 640×640 input
        if max_area < (16 * 16) / (640 * 640):
            buckets["very_small"].append(img_path)
        elif max_area < (32 * 32) / (640 * 640):
            buckets["small"].append(img_path)
        elif max_area < (64 * 64) / (640 * 640):
            buckets["medium"].append(img_path)
        else:
            buckets["large"].append(img_path)

    # Sample evenly: take 1-2 from each bucket up to n total
    per = max(1, n // 4)
    picked = []
    for group in ("very_small", "small", "medium", "large"):
        picked.extend(buckets[group][:per])
    # If we still need more, top up from buckets that have leftovers
    if len(picked) < n:
        for group in ("small", "medium", "very_small", "large"):
            for img in buckets[group][per:]:
                if len(picked) >= n:
                    break
                picked.append(img)
    return picked[:n]


def render_detections(runs, labels, runs_dir: Path, fig_dir: Path,
                      n_examples: int = 6, conf: float = 0.25):
    # Pick more candidates than needed so we can drop ones nobody detected
    candidates = _pick_drone_examples(n_examples * 3)
    if not candidates:
        print("[WARN] no test images with drone annotations"); return
    print(f"[INFO] starting with {len(candidates)} candidate drone images")

    out_root = fig_dir / "detections"
    out_root.mkdir(parents=True, exist_ok=True)

    # Run inference for every candidate and remember which images each
    # model actually produced a detection on. We'll then keep only images
    # where at least one of the four models drew a box.
    models = {}
    detected = {img: 0 for img in candidates}
    for run, label in zip(runs, labels):
        weights = runs_dir / run / "weights" / "best.pt"
        if not weights.exists():
            print(f"[WARN] missing {weights}"); continue
        model = YOLO(str(weights))
        models[run] = model
        run_out = out_root / run
        run_out.mkdir(exist_ok=True)
        for img_path in candidates:
            r = model.predict(img_path, conf=conf, verbose=False)[0]
            cv2.imwrite(str(run_out / Path(img_path).name), r.plot())
            if r.boxes is not None and len(r.boxes) > 0:
                detected[img_path] += 1
        print(f"[INFO] saved detections to {run_out}")

    # Keep images where at least one model detected a drone
    kept = [img for img in candidates if detected[img] >= 1][:n_examples]
    if not kept:
        kept = candidates[:n_examples]    # fallback — accept zero-detection rows
    print(f"[INFO] panel will use {len(kept)} of {len(candidates)} candidates")

    # composite panel
    n_runs = len(runs)
    n_rows = len(kept)
    fig, axes = plt.subplots(n_rows, n_runs,
                              figsize=(3.6 * n_runs, 3.6 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    if n_runs == 1:
        axes = axes.reshape(-1, 1)
    for i, img_path in enumerate(kept):
        for j, (run, label) in enumerate(zip(runs, labels)):
            p = out_root / run / Path(img_path).name
            if not p.exists():
                continue
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            axes[i, j].imshow(img); axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(label, fontsize=10)

    out = fig_dir / "fig_detections_panel.png"
    plt.tight_layout(); plt.savefig(out); plt.close()
    print(f"[INFO] saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs",   nargs="+", default=[r["name"] for r in RUNS])
    p.add_argument("--labels", nargs="+", default=LABELS)
    p.add_argument("--n-examples", type=int, default=6)
    args = p.parse_args()

    runs_dir    = get_runs_dir()
    results_dir = get_results_dir()
    fig_dir     = get_figures_dir()

    fig_recall_by_size(results_dir, fig_dir)
    fig_overall_metrics(results_dir, fig_dir)
    fig_training_curves(args.runs, args.labels, runs_dir, fig_dir)
    copy_pr_curves(args.runs, runs_dir, fig_dir)
    render_detections(args.runs, args.labels, runs_dir, fig_dir,
                      n_examples=args.n_examples)


if __name__ == "__main__":
    main()
