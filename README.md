# Drone Detection — YOLOv8s / YOLOv11s with Adaptive Area-Weighted Loss

Reference implementation for the experiment in
*"Adaptive loss function for improving detection of small-scale UAVs"*.

The adaptive loss multiplies each training sample's loss by

    w_i = min(A_0 / A_i, w_max),     A_0 = 32 × 32 = 1024 px,  w_max = 4

where `A_i` is the bounding-box area in pixels. Small targets receive a
larger gradient signal; losses for objects with `A_i ≥ A_0` are
unchanged. Implementation hooks the `TaskAlignedAssigner` so it works
across Ultralytics versions and propagates to both bbox and cls losses.

---

## One-command end-to-end run

| Environment | Default epochs | Output |
|---|---|---|
| Local (PyCharm CE, M3 Air) | **10** | `./runs/`, `./results/`, `./figures/` |
| Google Colab (T4 / L4 / A100) | **80** | `/content/drive/MyDrive/drone_yolo11/...` |

```bash
python experiment.py
```

Trains all 4 models, generates Tables 1–3 and all figures.

Override defaults if needed:
```bash
python experiment.py --epochs 30 --batch 8
python experiment.py --skip-train       # only evaluation + figures
```

---

## PyCharm CE (local)

```bash
git clone https://github.com/rskalish/drone-yolo11.git
cd drone-yolo11
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python experiment.py               # 10 epochs per model
```

Apple Silicon (M3 Air) uses MPS automatically. To prevent the Mac from
sleeping during training:

```bash
caffeinate -i python experiment.py
```

> 10 epochs on M3 Air: ~30–40 min per model, ~2–3 h for all 4.

---

## Google Colab

`Runtime → Change runtime type → A100` (or L4 / T4).

**Cell 1** — mount Drive and clone:
```python
from google.colab import drive
drive.mount("/content/drive", force_remount=True)
```

**Cell 2** — clone and install:
```bash
%cd /content
!rm -rf drone-yolo11
!git clone https://github.com/rskalish/drone-yolo11.git
%cd drone-yolo11
!pip install -r requirements.txt -q
```

**Cell 3** — run everything (saves directly to Drive):
```bash
!python experiment.py
```

If the Colab session disconnects, just re-run Cell 1 → 2 → 3. Training
auto-resumes from `last.pt` on Drive and skips already-completed runs.

---

## File layout

```
config.py             -- environment detection + shared constants
utils.py              -- label parsing, IoU, size groups, device pick
adaptive_loss.py      -- AdaptiveDetectionLoss + enable_adaptive_loss()
download_dataset.py   -- dataset fetch + train/valid/test split

train.py              -- training entry point (--all, --adaptive)
predict.py            -- inference on image / video / webcam
dataset_stats.py      -- Table 1 (size distribution)
evaluate_size.py      -- per-size Recall + localization error
compare_runs.py       -- aggregate Tables 2 and 3
figures.py            -- publication figures
experiment.py         -- orchestrator: runs everything end-to-end

requirements.txt
```

---

## Outputs

```
runs/<run_name>/                                 -- per-run artefacts
  weights/best.pt        weights/last.pt
  results.csv            results.png
  PR_curve.png           confusion_matrix.png

results/
  table1_distribution.csv     -- sample distribution by size group
  table2_overall.csv          -- P, R, F1, mAP@0.5, mAP@0.5:0.95
  table3_recall_by_size.csv   -- Recall per size group
  localization_quality.csv    -- mean centre error per size group
  <run>.json                  -- raw counts per run

figures/
  fig_recall_by_size.png      -- bar chart of Table 3
  fig_overall_metrics.png     -- bar chart of Table 2
  fig_training_curves.png     -- merged loss + mAP curves
  fig_detections_panel.png    -- side-by-side detection comparison
  detections/<run>/<img>.jpg  -- per-image annotated detections
  pr_curves/                  -- collected PR curves
```

---

## Metrics

* **Precision / Recall / F1** at IoU 0.5, conf 0.25
* **mAP@0.5** — mean Average Precision at a single IoU threshold of 0.5
* **mAP@0.5:0.95** — primary YOLO/COCO metric. Averages mAP across 10
  IoU thresholds (0.5, 0.55, …, 0.95). Stricter and more sensitive to
  localization quality than mAP@0.5.

---

## Single-run examples

```bash
# Train one configuration:
python train.py --model yolov8s.pt
python train.py --model yolov8s.pt --adaptive
python train.py --model yolo11s.pt --adaptive

# Evaluate per-size Recall:
python evaluate_size.py --weights runs/yolov8s_adaptive/weights/best.pt

# Inference:
python predict.py --source data/test/images --show
python predict.py --source path/to/video.mp4
python predict.py --source 0 --show               # webcam
```
