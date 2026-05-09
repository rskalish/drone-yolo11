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
DRIVE_RUNS = "/content/drive/MyDrive/drone_yolo11/runs"


def get_project_dir():
    """Write directly to Drive if running in Colab and Drive is mounted."""
    if os.path.isdir("/content/drive/MyDrive"):
        os.makedirs(DRIVE_RUNS, exist_ok=True)
        print(f"[INFO] Saving runs to Drive: {DRIVE_RUNS}")
        return DRIVE_RUNS
    return "runs"


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
    p.add_argument("--project",  default=None, help="save dir (auto: Drive in Colab, runs/ locally)")
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
    project = args.project or get_project_dir()
    is_resume = model_name.endswith("last.pt") and os.path.exists(model_name)

    model = YOLO(model_name)

    if is_resume:
        print(f"[INFO] Resuming from {model_name}")
        model.train(resume=True, device=device)
    else:
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
            project=project,
            name=run_name,
            exist_ok=True,
            device=device,
            plots=True,
            save=True,
            save_period=10,
        )

    best_path = os.path.join(project, run_name, "weights", "best.pt")
    print(f"\n[INFO] Training complete: {best_path}")

    from ultralytics import YOLO as _YOLO
    best = _YOLO(best_path)
    metrics = best.val(data=YAML_PATH, project=project,
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

    project = args.project or get_project_dir()

    if args.all:
        print("\n[INFO] Running all 4 experiments\n")
        for cfg in ALL_RUNS:
            # skip if already finished
            best_pt = os.path.join(project, cfg["name"], "weights", "best.pt")
            if os.path.exists(best_pt):
                print(f"[INFO] Skipping {cfg['name']} (best.pt already exists)")
                continue
            # resume from last.pt if exists
            last_pt = os.path.join(project, cfg["name"], "weights", "last.pt")
            model_arg = last_pt if os.path.exists(last_pt) else cfg["model"]
            print(f"\n>>> {cfg['name']} (adaptive={cfg['adaptive']}, model={model_arg})\n")
            train_one(model_arg, cfg["name"], cfg["adaptive"], args, device)
    else:
        name = args.name
        if name is None:
            stem = os.path.splitext(os.path.basename(args.model))[0]
            name = f"{stem}_{'adaptive' if args.adaptive else 'baseline'}"
        train_one(args.model, name, args.adaptive, args, device)


if __name__ == "__main__":
    main()
