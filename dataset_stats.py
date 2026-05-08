"""Compute Table 1 (sample distribution by size) for the article.

Walks data/{train,valid,test} and counts annotated boxes per size group:
very_small (<= 16x16), small (16-32), medium (32-64), large (>64).

Output: console table + CSV at results/table1_distribution.csv.
"""

import csv
import glob
import os
from pathlib import Path

import cv2

from evaluate_size import GROUPS, group_for_area, yolo_label_to_xyxy


def main():
    rows = []
    totals = {g[0]: 0 for g in GROUPS}
    overall_imgs = overall_objs = 0

    for split in ("train", "valid", "test"):
        img_dir = Path("data") / split / "images"
        lbl_dir = Path("data") / split / "labels"
        imgs = sorted(glob.glob(str(img_dir / "*.jpg")) +
                      glob.glob(str(img_dir / "*.png")))

        counts = {g[0]: 0 for g in GROUPS}
        n_objs = 0

        for img_path in imgs:
            lbl = lbl_dir / (Path(img_path).stem + ".txt")
            if not lbl.exists():
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            with open(lbl) as f:
                for line in f:
                    if not line.strip():
                        continue
                    box = yolo_label_to_xyxy(line, w, h)
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    counts[group_for_area(area)] += 1
                    n_objs += 1

        for g in counts:
            totals[g] += counts[g]
        overall_imgs += len(imgs)
        overall_objs += n_objs

        rows.append({
            "split":      split,
            "images":     len(imgs),
            "objects":    n_objs,
            "very_small": counts["very_small"],
            "small":      counts["small"],
            "medium":     counts["medium"],
            "large":      counts["large"],
        })

    rows.append({
        "split":      "total",
        "images":     overall_imgs,
        "objects":    overall_objs,
        "very_small": totals["very_small"],
        "small":      totals["small"],
        "medium":     totals["medium"],
        "large":      totals["large"],
    })

    print(f"\n{'split':<10} {'imgs':>6} {'objs':>7} "
          f"{'<=16':>7} {'16-32':>7} {'32-64':>7} {'>64':>7}")
    print("-" * 56)
    for r in rows:
        print(f"{r['split']:<10} {r['images']:>6} {r['objects']:>7} "
              f"{r['very_small']:>7} {r['small']:>7} "
              f"{r['medium']:>7} {r['large']:>7}")

    os.makedirs("results", exist_ok=True)
    out_csv = "results/table1_distribution.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[INFO] Saved {out_csv}")


if __name__ == "__main__":
    main()
