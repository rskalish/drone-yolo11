"""Train a YOLO detector on the drone dataset.

Examples:
    python train.py --model yolov8s.pt --name v8s_baseline
    python train.py --model yolov8s.pt --name v8s_adaptive --adaptive
    python train.py --model yolo11s.pt --name v11s_baseline
    python train.py --model yolo11s.pt --name v11s_adaptive --adaptive
"""

import argparse
import os

import torch

from download_dataset import download

YAML_PATH = os.path.join("data", "data.yaml")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    default="yolov8s.pt", help="pretrained weights (yolov8s.pt, yolo11s.pt, ...)")
    p.add_argument("--name",     default=None,         help="run name; default derived from model + loss")
    p.add_argument("--epochs",   type=int, default=80)
    p.add_argument("--imgsz",    type=int, default=640)
    p.add_argument("--batch",    type=int, default=16)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--project",  default="runs")
    p.add_argument("--adaptive", action="store_true", help="use adaptive area-weighted loss")
    p.add_argument("--a0",       type=float, default=32 * 32, help="reference area in pixels (default 1024)")
    p.add_argument("--w-max",    type=float, default=4.0,     help="max sample weight (default 4.0)")
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    if args.adaptive:
        from adaptive_loss import enable_adaptive_loss
        enable_adaptive_loss(a0=args.a0, w_max=args.w_max)

    from ultralytics import YOLO  # imported AFTER monkey-patch

    download()

    if torch.cuda.is_available():
        device = 0
        print(f"[INFO] Device: GPU {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("[INFO] Device: Apple MPS")
    else:
        device = "cpu"
        print("[INFO] Device: CPU")

    if args.name is None:
        stem = os.path.splitext(os.path.basename(args.model))[0]
        suffix = "adaptive" if args.adaptive else "baseline"
        args.name = f"{stem}_{suffix}"

    model = YOLO(args.model)

    model.train(
        data=YAML_PATH,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        lr0=0.01,
        lrf=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        mosaic=1.0,
        fliplr=0.5,
        degrees=5.0,
        scale=0.5,
        seed=args.seed,
        deterministic=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
        device=device,
        plots=True,
        save=True,
    )

    best_path = os.path.join(args.project, args.name, "weights", "best.pt")
    print(f"\n[INFO] Training complete. Best model: {best_path}")

    print("[INFO] Validating...")
    best = YOLO(best_path)
    metrics = best.val(data=YAML_PATH, project=args.project, name=f"{args.name}_val", exist_ok=True)

    print(f"\n{'=' * 40}")
    print(f"Run: {args.name}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    main()
