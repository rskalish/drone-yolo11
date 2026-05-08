import os
import argparse
from ultralytics import YOLO

DEFAULT_MODEL  = os.path.join("drone_detection", "yolo11s_run1", "weights", "best.pt")
DEFAULT_SOURCE = os.path.join("data", "test", "images")


def main():
    parser = argparse.ArgumentParser(description="Drone detection inference")
    parser.add_argument("--model",  default=DEFAULT_MODEL,  help="Path to best.pt")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Folder/file/video/0 (webcam)")
    parser.add_argument("--conf",   type=float, default=0.25)
    parser.add_argument("--iou",    type=float, default=0.45)
    parser.add_argument("--show",   action="store_true", help="Show results in a window")
    args = parser.parse_args()

    model = YOLO(args.model)

    model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save=True,
        show=args.show,
        project="predictions",
        name="run",
        exist_ok=True,
    )

    print("[INFO] Results saved to predictions/run/")


if __name__ == "__main__":
    main()
