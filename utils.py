"""Shared helpers: label parsing, IoU, size grouping, device selection."""

import numpy as np
import torch

from config import SIZE_GROUPS


def yolo_label_to_xyxy(line: str, w: int, h: int) -> np.ndarray:
    """Convert one line of a YOLO label file to absolute xyxy in pixels."""
    parts = line.strip().split()
    cx, cy, bw, bh = map(float, parts[1:5])
    return np.array([
        (cx - bw / 2) * w,
        (cy - bh / 2) * h,
        (cx + bw / 2) * w,
        (cy + bh / 2) * h,
    ], dtype=np.float32)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two xyxy boxes."""
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def group_for_area(area: float) -> str:
    """Map a bounding-box area (px^2) to a size-group name."""
    for name, lo, hi in SIZE_GROUPS:
        if lo <= area < hi:
            return name
    return "large"


def get_device():
    """Return the best available device: 0 (CUDA), 'mps', or 'cpu'."""
    if torch.cuda.is_available():
        print(f"[INFO] Device: GPU {torch.cuda.get_device_name(0)}")
        return 0
    if torch.backends.mps.is_available():
        print("[INFO] Device: Apple MPS")
        return "mps"
    print("[INFO] Device: CPU")
    return "cpu"
