import os
import torch
from ultralytics import YOLO
from download_dataset import download

YAML_PATH    = os.path.join("data", "data.yaml")
MODEL_SIZE   = "yolo11s"       # n / s / m / l / x
EPOCHS       = 100
IMGSZ        = 640
BATCH        = 8               # зменш до 4 якщо мало VRAM
PATIENCE     = 20
PROJECT      = "drone_detection"
RUN_NAME     = f"{MODEL_SIZE}_run1"


def main():
    download()

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Пристрій: {'GPU ' + torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    model = YOLO(f"{MODEL_SIZE}.pt")

    model.train(
        data=YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        lr0=0.01,
        lrf=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        mosaic=1.0,
        fliplr=0.5,
        degrees=5.0,
        scale=0.5,
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,
        device=device,
        plots=True,
        save=True,
    )

    best_path = os.path.join(PROJECT, RUN_NAME, "weights", "best.pt")
    print(f"\n[INFO] Тренування завершено. Найкраща модель: {best_path}")

    print("[INFO] Валідація...")
    best = YOLO(best_path)
    metrics = best.val(data=YAML_PATH)

    print(f"\n{'='*40}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision:{metrics.box.mp:.4f}")
    print(f"Recall:   {metrics.box.mr:.4f}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
