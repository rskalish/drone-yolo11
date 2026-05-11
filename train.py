"""Train a YOLO detector on the drone dataset.

Defaults adapt to the runtime:
    Colab (GPU): 80 epochs, batch 16
    Local:       10 epochs, batch 8

Examples:
    python train.py --all                       # all 4 experiments
    python train.py --model yolov8s.pt          # one run, baseline
    python train.py --model yolov8s.pt --adaptive
"""

import argparse
import os

from config import (
    DATASET_YAML_V8,
    DATASET_YAML_V11,
    DEFAULT_BATCH,
    DEFAULT_EPOCHS,
    DEFAULT_PATIENCE,
    RUNS,
    A0,
    W_MAX,
    env_summary,
    get_runs_dir,
)
from download_dataset import download_all
from utils import get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--all",      action="store_true", help="run all 4 experiments")
    p.add_argument("--model",    default="yolov8s.pt")
    p.add_argument("--name",     default=None)
    p.add_argument("--epochs",   type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--imgsz",    type=int,   default=640)
    p.add_argument("--batch",    type=int,   default=DEFAULT_BATCH)
    p.add_argument("--patience", type=int,   default=DEFAULT_PATIENCE)
    p.add_argument("--project",  default=None, help="output dir (auto-detected)")
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--a0",       type=float, default=A0)
    p.add_argument("--w-max",    type=float, default=W_MAX)
    p.add_argument("--seed",     type=int,   default=42)
    # dataset override for single-run mode
    p.add_argument("--data",     default=None, help="path to data.yaml (auto if omitted)")
    return p.parse_args()


def train_one(model_name: str, run_name: str, adaptive: bool,
              dataset_yaml: str, args, device, project):
    """Train one configuration.

    IMPORTANT: enable/disable adaptive loss BEFORE importing YOLO so the
    monkey-patch is in place when DetectionModel.init_criterion is first called.
    adaptive=True  → enable_adaptive_loss()  (patches DetectionModel)
    adaptive=False → disable_adaptive_loss() (restores original criterion)
    This ensures sequential runs in --all mode don't bleed patches across models.
    """
    from adaptive_loss import disable_adaptive_loss, enable_adaptive_loss

    if adaptive:
        enable_adaptive_loss(a0=args.a0, w_max=args.w_max)
    else:
        disable_adaptive_loss()

    from ultralytics import YOLO

    is_resume = model_name.endswith("last.pt") and os.path.exists(model_name)
    model = YOLO(model_name)

    if is_resume:
        print(f"[INFO] Resuming from {model_name}")
        model.train(resume=True, device=device)
    else:
        model.train(
            data=dataset_yaml,
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
            project=str(project),
            name=run_name,
            exist_ok=True,
            device=device,
            plots=True,
            save=True,
            save_period=10,
        )

    best_path = project / run_name / "weights" / "best.pt"
    print(f"\n[INFO] Training complete: {best_path}")

    from ultralytics import YOLO as _YOLO
    metrics = _YOLO(str(best_path)).val(
        data=dataset_yaml,
        project=str(project),
        name=f"{run_name}_val",
        exist_ok=True,
    )
    print(f"\n{'=' * 40}")
    print(f"Run:       {run_name}")
    print(f"Adaptive:  {adaptive}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"{'=' * 40}\n")


def main():
    args = parse_args()
    print(env_summary())
    download_all()
    device  = get_device()
    project = args.project or get_runs_dir()

    if args.all:
        print(f"\n[INFO] Running all {len(RUNS)} experiments\n")
        for cfg in RUNS:
            best_pt = project / cfg["name"] / "weights" / "best.pt"
            if best_pt.exists():
                print(f"[INFO] Skipping {cfg['name']} (best.pt already exists)")
                continue
            last_pt = project / cfg["name"] / "weights" / "last.pt"
            model_arg = str(last_pt) if last_pt.exists() else cfg["model"]
            dataset_yaml = str(cfg["dataset_yaml"])
            print(f"\n>>> {cfg['name']}  adaptive={cfg['adaptive']}  "
                  f"model={model_arg}  data={dataset_yaml}\n")
            train_one(model_arg, cfg["name"], cfg["adaptive"],
                      dataset_yaml, args, device, project)
    else:
        # Single-run mode: infer dataset from model name if not given
        if args.data:
            dataset_yaml = args.data
        elif "v8" in args.model or "yolov8" in args.model.lower():
            dataset_yaml = str(DATASET_YAML_V8)
        else:
            dataset_yaml = str(DATASET_YAML_V11)

        name = args.name
        if name is None:
            stem = os.path.splitext(os.path.basename(args.model))[0]
            name = f"{stem}_{'adaptive' if args.adaptive else 'baseline'}"
        train_one(args.model, name, args.adaptive,
                  dataset_yaml, args, device, project)


if __name__ == "__main__":
    main()
