"""Per-size-group Recall evaluation.

For each test image: match predictions to GT boxes (greedy IoU >= 0.5),
classify each GT into a size group by area, count TP/FN, compute Recall
and the mean centre-distance error normalized by image diagonal.

Usage:
    python evaluate_size.py --weights runs/yolov8s_baseline/weights/best.pt \\
                            --out results/yolov8s_baseline.json
"""

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from config import DATA_DIR, SIZE_GROUPS, get_results_dir
from utils import group_for_area, iou_xyxy, yolo_label_to_xyxy


def evaluate(weights: str, split: str, imgsz: int, conf: float, iou_thr: float):
    images_dir = DATA_DIR / split / "images"
    labels_dir = DATA_DIR / split / "labels"
    image_paths = sorted(glob.glob(str(images_dir / "*.jpg")) +
                         glob.glob(str(images_dir / "*.png")))
    print(f"[INFO] {len(image_paths)} images in {split}")

    model = YOLO(weights)
    stats = {g[0]: {"tp": 0, "fn": 0, "fp": 0, "centre_err": [], "n_gt": 0}
             for g in SIZE_GROUPS}

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        diag = (h ** 2 + w ** 2) ** 0.5

        # ground truths
        lbl_path = labels_dir / (Path(img_path).stem + ".txt")
        gts = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                gts = [yolo_label_to_xyxy(line, w, h)
                       for line in f if line.strip()]

        # predictions
        result = model.predict(img_path, imgsz=imgsz, conf=conf, verbose=False)[0]
        preds = (result.boxes.xyxy.cpu().numpy()
                 if result.boxes is not None else np.zeros((0, 4)))

        # match GT → prediction (greedy by IoU)
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

            if best_iou >= iou_thr:
                stats[grp]["tp"] += 1
                matched_pred.add(best_j)
                gt_c = np.array([(gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2])
                pr = preds[best_j]
                pr_c = np.array([(pr[0] + pr[2]) / 2, (pr[1] + pr[3]) / 2])
                stats[grp]["centre_err"].append(
                    float(np.linalg.norm(gt_c - pr_c) / diag))
            else:
                stats[grp]["fn"] += 1

        for j, pred in enumerate(preds):
            if j in matched_pred:
                continue
            area = (pred[2] - pred[0]) * (pred[3] - pred[1])
            stats[group_for_area(area)]["fp"] += 1

    # finalize stats
    out = {}
    overall = {"tp": 0, "fn": 0, "fp": 0, "errs": []}
    for name, _, _ in SIZE_GROUPS:
        s = stats[name]
        rec  = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else 0.0
        prec = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        out[name] = {
            "n_gt":    s["n_gt"],
            "tp":      s["tp"],
            "fn":      s["fn"],
            "fp":      s["fp"],
            "recall":    round(rec, 4),
            "precision": round(prec, 4),
            "f1":        round(f1, 4),
            "centre_err_mean": (round(float(np.mean(s["centre_err"])), 5)
                                if s["centre_err"] else None),
        }
        overall["tp"] += s["tp"]; overall["fn"] += s["fn"]; overall["fp"] += s["fp"]
        overall["errs"].extend(s["centre_err"])

    rec  = overall["tp"] / (overall["tp"] + overall["fn"]) if (overall["tp"] + overall["fn"]) else 0.0
    prec = overall["tp"] / (overall["tp"] + overall["fp"]) if (overall["tp"] + overall["fp"]) else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    out["overall"] = {
        "tp": overall["tp"], "fn": overall["fn"], "fp": overall["fp"],
        "recall": round(rec, 4), "precision": round(prec, 4), "f1": round(f1, 4),
        "centre_err_mean": (round(float(np.mean(overall["errs"])), 5)
                            if overall["errs"] else None),
    }
    out["meta"] = {
        "weights": weights, "split": split, "conf": conf,
        "iou_thr": iou_thr, "imgsz": imgsz,
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--split",   default="test", choices=["train", "valid", "test"])
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--out",     default=None)
    args = p.parse_args()

    out = evaluate(args.weights, args.split, args.imgsz, args.conf, args.iou_thr)
    print(json.dumps(out, indent=2))

    if args.out:
        out_path = Path(args.out)
    else:
        run_name = Path(args.weights).parent.parent.name
        out_path = get_results_dir() / f"{run_name}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[INFO] Saved {out_path}")


if __name__ == "__main__":
    main()
