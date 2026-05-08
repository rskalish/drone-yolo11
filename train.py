"""Train a YOLO detector on the drone dataset.

Examples:
    # Run all 4 experiments at once (for the article):
    python train.py --all

    # Single run:
    python train.py --model yolov8s.pt
    python train.py --model yolov8s.pt --adaptive
    python train.py --model yolo11s.pt
    python train.py --model yolo11s.pt --adaptive
"""

import argparse
import os

import torch

from download_dataset import download

YAML_PATH = os.path.join("data", "data.yaml")

ALL_RUNS = [
    {"model": "yolov8s.pt", "name": "yolov8s_baseline", "adaptive": False},
    {"model": "yolov8s.pt", "name": "yolov8s_adaptive",  "adaptive": True},
    {"model": "yolo11s.pt", "name": "yolo11s_baseline",  "adaptive": False},
    {"model": "yolo11s.pt", "name": "yolo11s_adaptive",  "adaptive": True},
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--all",      action="store_true", help="run all 4 experiments (for the article)")
    p.add_argument("--model",    default="yolov8s.pt")
    p.add_argument("--name",     default=None)
    p.add_argument("--epochs",   type=int,   default=80)
    p.add_argument("--imgsz",    type=int,   default=640)
    p.add_argument("--batch",    type=int,   default=16)
    p.add_argument("--patience", type=int,   default=20)
    p.add_argument("--project",  default="runs")
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--a0",       type=float, default=32 * 32)
    p.add_argument("--w-max",    type=float, default=4.0)
    p.add_argument("--seed",     type=int,   default=42)
    return p.parse_args()


def get_device():
    if torch.cuda.is_available():
        print(f"[INFO] Device: GPU {torch.cuda.get_device_name(0)}")
        return 0
    elif torch.backends.mps.is_available():
        print("[INFO] Device: Apple MPS")
        return "mps"
    else:
        print("[INFO] Device: CPU")
        return "cpu"


def train_one(model_name, run_name, adaptive, args, device):
    if adaptive:
        from adaptive_loss import enable_adaptive_loss
        enable_adaptive_loss(a0=args.a0, w_max=args.w_max)

    from ultralytics import YOLO
    model = YOLO(model_name)
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
        name=run_name,
        exist_ok=True,
        device=device,
        plots=True,
        save=True,
    )

    best_path = os.path.join(args.project, run_name, "weights", "best.pt")
    print(f"\n[INFO] Training complete: {best_path}")

    from ultralytics import YOLO as _YOLO
    best = _YOLO(best_path)
    metrics = best.val(data=YAML_PATH, project=args.project,
                       name=f"{run_name}_val", exist_ok=True)
    print(f"\n{'=' * 40}")
    print(f"Run:       {run_name}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"{'=' * 40}\n")


def main():
    args = parse_args()
    download()
    device = get_device()

    if args.all:
        print("\n[INFO] Running all 4 experiments\n")
        for cfg in ALL_RUNS:
            print(f"\n>>> {cfg['name']} (adaptive={cfg['adaptive']})\n")
            train_one(cfg["model"], cfg["name"], cfg["adaptive"], args, device)
    else:
        name = args.name
        if name is None:
            stem = os.path.splitext(os.path.basename(args.model))[0]
            name = f"{stem}_{'adaptive' if args.adaptive else 'baseline'}"
        train_one(args.model, name, args.adaptive, args, device)


if __name__ == "__main__":
    main()
