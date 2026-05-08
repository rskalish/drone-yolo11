import os
import glob
import shutil
import random
import yaml
import zipfile
import requests

DATASET_URL = "https://app.roboflow.com/ds/3B4hCrEJ8A?key=ymcXaw0qAq"
DATASET_ZIP = "dataset.zip"
DATASET_RAW = "dataset_raw"
DATASET_DIR = "data"


def download():
    if os.path.exists(DATASET_DIR):
        print(f"[INFO] Dataset already exists: {DATASET_DIR}/")
        return

    if not os.path.exists(DATASET_ZIP):
        print("[INFO] Downloading dataset...")
        response = requests.get(DATASET_URL, stream=True, allow_redirects=True)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        received = 0
        with open(DATASET_ZIP, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                received += len(chunk)
                if total:
                    print(f"\r[INFO] {received / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB", end="")
        print("\n[INFO] Download complete.")

    print("[INFO] Extracting...")
    with zipfile.ZipFile(DATASET_ZIP, "r") as z:
        z.extractall(DATASET_RAW)

    _split_dataset()
    print(f"[INFO] Dataset ready: {DATASET_DIR}/")


def _split_dataset(train_ratio=0.70, val_ratio=0.20):
    random.seed(42)

    src_images = os.path.join(DATASET_RAW, "train", "images")
    src_labels = os.path.join(DATASET_RAW, "train", "labels")

    all_images = sorted(
        glob.glob(os.path.join(src_images, "*.jpg")) +
        glob.glob(os.path.join(src_images, "*.png"))
    )
    random.shuffle(all_images)

    n = len(all_images)
    n_val   = int(n * val_ratio)
    n_test  = int(n * (1 - train_ratio - val_ratio))
    n_train = n - n_val - n_test

    splits = {
        "train": all_images[:n_train],
        "valid": all_images[n_train:n_train + n_val],
        "test":  all_images[n_train + n_val:],
    }

    print(f"[INFO] Split: train={n_train} | valid={n_val} | test={n_test}")

    for split, imgs in splits.items():
        os.makedirs(os.path.join(DATASET_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, split, "labels"), exist_ok=True)
        for img_path in imgs:
            fname = os.path.basename(img_path)
            stem  = os.path.splitext(fname)[0]
            shutil.copy(img_path, os.path.join(DATASET_DIR, split, "images", fname))
            lbl = os.path.join(src_labels, stem + ".txt")
            if os.path.exists(lbl):
                shutil.copy(lbl, os.path.join(DATASET_DIR, split, "labels", stem + ".txt"))

    cfg = {
        "path":  os.path.abspath(DATASET_DIR),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    1,
        "names": ["drones"],
    }
    with open(os.path.join(DATASET_DIR, "data.yaml"), "w") as f:
        yaml.dump(cfg, f)


if __name__ == "__main__":
    download()
