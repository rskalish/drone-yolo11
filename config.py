"""Shared configuration: environment detection, paths, run definitions."""

import os
from pathlib import Path

# ── Environment detection ────────────────────────────────────────────────────
IS_COLAB = "COLAB_GPU" in os.environ or os.path.isdir("/content")
IS_LOCAL = not IS_COLAB

# ── Defaults that differ between Colab and local ─────────────────────────────
DEFAULT_EPOCHS   = 80 if IS_COLAB else 10
DEFAULT_BATCH    = 16 if IS_COLAB else 8
DEFAULT_PATIENCE = 20 if IS_COLAB else 10

# ── Dataset paths (separate datasets per YOLO family) ────────────────────────
DATA_DIR_V8  = Path("data_v8")   # YOLOv8 format dataset
DATA_DIR_V11 = Path("data_v11")  # YOLOv11 format dataset

DATASET_YAML_V8  = DATA_DIR_V8  / "data.yaml"
DATASET_YAML_V11 = DATA_DIR_V11 / "data.yaml"

# Legacy alias (used by dataset_stats, evaluate_size when called standalone)
DATA_DIR     = DATA_DIR_V11
DATASET_YAML = DATASET_YAML_V11

# ── Roboflow dataset URLs ─────────────────────────────────────────────────────
DATASET_URL_V8  = "https://app.roboflow.com/ds/RUtAN730lu?key=rn5noQXlBB"
DATASET_URL_V11 = "https://app.roboflow.com/ds/fprdklA54R?key=OI4DskY6Kp"

# ── Drive / local paths ───────────────────────────────────────────────────────
DRIVE_ROOT = Path("/content/drive/MyDrive/drone_yolo11")
LOCAL_RUNS = Path("runs")


def get_runs_dir() -> Path:
    """Save runs to Google Drive in Colab (if mounted), else ./runs/."""
    if IS_COLAB and Path("/content/drive/MyDrive").is_dir():
        runs = DRIVE_ROOT / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        return runs
    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
    return LOCAL_RUNS


def get_results_dir() -> Path:
    """Tables and JSON metrics output directory."""
    if IS_COLAB and Path("/content/drive/MyDrive").is_dir():
        d = DRIVE_ROOT / "results"
    else:
        d = Path("results")
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_figures_dir() -> Path:
    """Article figures output directory."""
    if IS_COLAB and Path("/content/drive/MyDrive").is_dir():
        d = DRIVE_ROOT / "figures"
    else:
        d = Path("figures")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Experiment definitions ────────────────────────────────────────────────────
# Each run specifies which dataset YAML to use so models are always evaluated
# on their own dataset format.
RUNS = [
    {"model": "yolov8s.pt",  "name": "yolov8s_baseline", "adaptive": False, "dataset_yaml": DATASET_YAML_V8},
    {"model": "yolov8s.pt",  "name": "yolov8s_adaptive",  "adaptive": True,  "dataset_yaml": DATASET_YAML_V8},
    {"model": "yolo11s.pt",  "name": "yolo11s_baseline", "adaptive": False, "dataset_yaml": DATASET_YAML_V11},
    {"model": "yolo11s.pt",  "name": "yolo11s_adaptive",  "adaptive": True,  "dataset_yaml": DATASET_YAML_V11},
]

LABELS = [
    "YOLOv8s baseline",
    "YOLOv8s + adaptive loss",
    "YOLOv11s baseline",
    "YOLOv11s + adaptive loss",
]

# ── Adaptive loss hyperparameters ─────────────────────────────────────────────
A0    = 32.0 * 32.0  # reference small-object area in pixels
W_MAX = 4.0          # max sample weight cap

# ── Size-group thresholds (pixels²) ──────────────────────────────────────────
SIZE_GROUPS = [
    ("very_small", 0,         16 * 16),
    ("small",      16 * 16,   32 * 32),
    ("medium",     32 * 32,   64 * 64),
    ("large",      64 * 64,   float("inf")),
]


def env_summary() -> str:
    return (
        f"Environment: {'Colab' if IS_COLAB else 'Local'} | "
        f"runs → {get_runs_dir()} | "
        f"defaults: epochs={DEFAULT_EPOCHS}, batch={DEFAULT_BATCH}"
    )
