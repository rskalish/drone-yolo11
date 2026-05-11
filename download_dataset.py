"""Download and prepare both datasets (YOLOv8 and YOLOv11 formats).

Roboflow exports may already contain train/valid/test splits.
If only a train/ folder is present the script splits it 70/20/10.
A data.yaml is written (or fixed) so paths are absolute.

Usage:
    python download_dataset.py          # downloads both datasets
    python download_dataset.py --v8     # YOLOv8 dataset only
    python download_dataset.py --v11    # YOLOv11 dataset only
"""

import argparse
import glob
import os
import random
import shutil
import zipfile
from pathlib import Path

import requests
import yaml

from config import (
    DATA_DIR_V8,
    DATA_DIR_V11,
    DATASET_URL_V8,
    DATASET_URL_V11,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _download_zip(url: str, dest_zip: str):
    """Stream-download url → dest_zip with a progress indicator."""
    print(f"[INFO] Downloading {url}")
    resp = requests.get(url, stream=True, allow_redirects=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    received = 0
    with open(dest_zip, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            received += len(chunk)
            if total:
                print(f"\r  {received / 1_048_576:.1f} / {total / 1_048_576:.1f} MB", end="", flush=True)
    print(f"\n[INFO] Saved {dest_zip}")


def _extract(zip_path: str, out_dir: str):
    print(f"[INFO] Extracting → {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


def _find_split_root(raw_dir: str) -> str:
    """Return the directory that contains train/, valid/ or test/ sub-folders."""
    for root, dirs, _ in os.walk(raw_dir):
        if any(d in dirs for d in ("train", "valid", "test")):
            return root
    return raw_dir


def _manual_split(src_images: str, src_labels: str, data_dir: Path,
                  train_ratio=0.70, val_ratio=0.20):
    """Split a single 'train' folder into train / valid / test."""
    random.seed(42)
    imgs = sorted(
        glob.glob(os.path.join(src_images, "*.jpg")) +
        glob.glob(os.path.join(src_images, "*.png"))
    )
    random.shuffle(imgs)
    n = len(imgs)
    n_val  = int(n * val_ratio)
    n_test = int(n * (1 - train_ratio - val_ratio))
    n_train = n - n_val - n_test
    splits = {
        "train": imgs[:n_train],
        "valid": imgs[n_train:n_train + n_val],
        "test":  imgs[n_train + n_val:],
    }
    print(f"[INFO] Manual split: train={n_train} | valid={n_val} | test={n_test}")
    for split, split_imgs in splits.items():
        (data_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (data_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        for img_path in split_imgs:
            fname = os.path.basename(img_path)
            stem  = os.path.splitext(fname)[0]
            shutil.copy(img_path, data_dir / split / "images" / fname)
            lbl = os.path.join(src_labels, stem + ".txt")
            if os.path.exists(lbl):
                shutil.copy(lbl, data_dir / split / "labels" / (stem + ".txt"))


def _copy_existing_splits(split_root: str, data_dir: Path):
    """Copy pre-split train/valid/test folders into data_dir."""
    for split in ("train", "valid", "test"):
        src = os.path.join(split_root, split)
        dst = data_dir / split
        if os.path.isdir(src):
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            n = len(list((dst / "images").glob("*"))) if (dst / "images").exists() else 0
            print(f"[INFO]   {split}: {n} images")


def _write_yaml(data_dir: Path, nc: int = 1, names: list = None):
    """Write (or overwrite) data.yaml with absolute paths."""
    if names is None:
        names = ["drones"]
    cfg = {
        "path":  str(data_dir.resolve()),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    nc,
        "names": names,
    }
    out = data_dir / "data.yaml"
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"[INFO] Written {out}")


def _infer_classes(split_root: str) -> tuple[int, list]:
    """Try to read nc / names from the extracted data.yaml."""
    yaml_path = os.path.join(split_root, "data.yaml")
    if not os.path.exists(yaml_path):
        return 1, ["drones"]
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    nc    = cfg.get("nc", 1)
    names = cfg.get("names", ["drones"])
    return nc, names


# ── main download + prepare ───────────────────────────────────────────────────

def prepare_dataset(url: str, data_dir: Path, label: str):
    """Download, extract and organise one dataset into data_dir."""
    if data_dir.exists() and (data_dir / "data.yaml").exists():
        print(f"[INFO] {label} dataset already exists: {data_dir}/")
        return

    zip_path = f"{data_dir.name}.zip"
    raw_dir  = f"{data_dir.name}_raw"

    # 1. Download
    if not os.path.exists(zip_path):
        _download_zip(url, zip_path)

    # 2. Extract
    if not os.path.isdir(raw_dir):
        _extract(zip_path, raw_dir)

    # 3. Locate split root
    split_root = _find_split_root(raw_dir)
    nc, names  = _infer_classes(split_root)
    has_valid  = os.path.isdir(os.path.join(split_root, "valid"))
    has_test   = os.path.isdir(os.path.join(split_root, "test"))

    data_dir.mkdir(parents=True, exist_ok=True)

    if has_valid and has_test:
        # Dataset already split by Roboflow
        print(f"[INFO] {label}: pre-split dataset detected, copying…")
        _copy_existing_splits(split_root, data_dir)
    else:
        # Only train/ present — split manually
        print(f"[INFO] {label}: no valid/test found, splitting train 70/20/10…")
        src_images = os.path.join(split_root, "train", "images")
        src_labels = os.path.join(split_root, "train", "labels")
        _manual_split(src_images, src_labels, data_dir)

    _write_yaml(data_dir, nc=nc, names=names)

    # Clean up zip and raw folder
    try:
        os.remove(zip_path)
        shutil.rmtree(raw_dir)
    except Exception:
        pass

    print(f"[INFO] {label} dataset ready: {data_dir}/")


def download_all():
    prepare_dataset(DATASET_URL_V8,  DATA_DIR_V8,  "YOLOv8")
    prepare_dataset(DATASET_URL_V11, DATA_DIR_V11, "YOLOv11")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--v8",  action="store_true", help="download YOLOv8 dataset only")
    p.add_argument("--v11", action="store_true", help="download YOLOv11 dataset only")
    args = p.parse_args()

    if args.v8:
        prepare_dataset(DATASET_URL_V8,  DATA_DIR_V8,  "YOLOv8")
    elif args.v11:
        prepare_dataset(DATASET_URL_V11, DATA_DIR_V11, "YOLOv11")
    else:
        download_all()
