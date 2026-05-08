"""Per-size-group Recall evaluation.

Splits ground-truth boxes into three categories by pixel area at the
inference resolution and computes Recall in each:

    very_small : <= 16 x 16 px
    small      : 16-32 px
    medium     : 32-64 px
    large      : > 64 px (ignored in tables but reported)

Output: JSON file with TP / FN / Recall per group, plus localization
quality (mean centre-distance error normalized by image diagonal).

Usage:
    python evaluate_size.py --weights runs/yolov8s_baseline/weights/best.pt \\
                            --split test --out results/v8s_baseline.json
"""

import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

GROUPS = [
    ("very_small", 0,    16 * 16),
    ("small",      16 * 16, 32 * 32),
    ("medium",     32 * 32, 64 * 64),
    ("large",      64 * 64, float("inf")),
]


def yolo_label_to_xyxy(line: str, w: int, h: int):
    parts = line.strip().split()
    cx, cy, bw, bh = map(float, parts[1:5])
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    inter_w = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    inter = inter_w * inter_h
    union = (xa2 - xa1) * (ya2 - ya1) + (xb2 - xb1) * (yb2 - yb1) - inter
    return inter / union if union > 0 else 0.0


def group_for_area(area: float) -> str:
    for name, lo, hi in GROUPS:
        if lo <= area < hi:
            return name
    return "large"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--split",   default="test", choices=["train", "valid", "test"])
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--data-root", default="data")
    p.add_argument("--out",     default=None)
    args = p.parse_args()

    images_dir = Path(args.data_root) / args.split / "images"
    labels_dir = Path(args.data_root) / args.split / "labels"
    image_paths = sorted(glob.glob(str(images_dir / "*.jpg")) +
                         glob.glob(str(images_dir / "*.png")))
    print(f"[INFO] {len(image_paths)} images in {args.split}")

    model = YOLO(args.weights)

    stats = {g[0]: {"tp": 0, "fn": 0, "fp": 0, "centre_err": [], "n_gt": 0}
             for g in GROUPS}

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        diag = (h ** 2 + w ** 2) ** 0.5

        lbl_path = labels_dir / (Path(img_path).stem + ".txt")
        gts = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    if line.strip():
                        gts.append(yolo_label_to_xyxy(line, w, h))

        # rescale GT to imgsz space (model resizes input internally;
        # we keep pixel coordinates in original image space and let YOLO
        # produce predictions in the same space - that is the default)
        result = model.predict(img_path, imgsz=args.imgsz, conf=args.conf,
                               verbose=False)[0]
        preds = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.zeros((0, 4))

        matched_pred = set()
        for gt in gts:
            area = (gt[2] - gt[0]) * (gt[3] - gt[1])
            grp = group_for_area(area)
            stats[grp]["n_gt"] += 1

            best_iou, best_j = 0.0, -1
            for j, pred in enumerate(preds):
                if j in matched_pred:
                    continue
                iou = iou_xyxy(gt, pred)
                if iou > best_iou:
                    best_iou, best_j = iou, j

            if best_iou >= args.iou_thr:
                stats[grp]["tp"] += 1
                matched_pred.add(best_j)
                # localization error
                gt_c = np.array([(gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2])
                pr = preds[best_j]
                pr_c = np.array([(pr[0] + pr[2]) / 2, (pr[1] + pr[3]) / 2])
                err = float(np.linalg.norm(gt_c - pr_c) / diag)
                stats[grp]["centre_err"].append(err)
            else:
                stats[grp]["fn"] += 1

        # any preds not matched are FPs (categorized by their own area)
        for j, pred in enumerate(preds):
            if j in matched_pred:
                continue
            area = (pred[2] - pred[0]) * (pred[3] - pred[1])
            grp = group_for_area(area)
            stats[grp]["fp"] += 1

    # finalize
    out = {}
    overall_tp = overall_fn = overall_fp = 0
    overall_errs: list[float] = []
    for name, _, _ in GROUPS:
        s = stats[name]
        recall = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else 0.0
        precision = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        out[name] = {
            "n_gt":      s["n_gt"],
            "tp":        s["tp"],
            "fn":        s["fn"],
            "fp":        s["fp"],
            "recall":    round(recall, 4),
            "precision": round(precision, 4),
            "f1":        round(f1, 4),
            "centre_err_mean": round(float(np.mean(s["centre_err"])), 5) if s["centre_err"] else None,
        }
        overall_tp += s["tp"]; overall_fn += s["fn"]; overall_fp += s["fp"]
        overall_errs.extend(s["centre_err"])

    rec = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) else 0.0
    prec = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    out["overall"] = {
        "tp": overall_tp, "fn": overall_fn, "fp": overall_fp,
        "recall":    round(rec, 4),
        "precision": round(prec, 4),
        "f1":        round(f1, 4),
        "centre_err_mean": round(float(np.mean(overall_errs)), 5) if overall_errs else None,
    }
    out["meta"] = {
        "weights": args.weights, "split": args.split,
        "conf": args.conf, "iou_thr": args.iou_thr, "imgsz": args.imgsz,
    }

    print(json.dumps(out, indent=2))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[INFO] Saved {args.out}")


if __name__ == "__main__":
    main()
