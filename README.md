# Drone Detection — YOLOv8s / YOLOv11s with Adaptive Area-Weighted Loss

Reference implementation for the experiment in
*"Adaptive loss function for improving detection of small-scale UAVs"*.

The adaptive loss multiplies each training sample's loss by
`w_i = min(A_0 / A_i, w_max)`, where `A_i` is the bounding box area in
pixels, `A_0 = 32 × 32 = 1024`, `w_max = 4`. Small targets receive a
larger gradient signal, while losses for objects with `A_i ≥ A_0` are
unchanged.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The dataset (DUT Anti-UAV style, 1 class `drones`) is downloaded and
split into `train / valid / test` automatically on first run.

## Reproducing the article experiments

The 4 runs below reproduce Tables 2 and 3 of the article.

```bash
# Baselines
python train.py --model yolov8s.pt --name yolov8s_baseline
python train.py --model yolo11s.pt --name yolo11s_baseline

# Adaptive loss
python train.py --model yolov8s.pt --name yolov8s_adaptive --adaptive
python train.py --model yolo11s.pt --name yolo11s_adaptive --adaptive
```

Default training matches the article: 80 epochs, imgsz 640, batch 16,
seed 42, `A_0 = 1024`, `w_max = 4`.

## Computing tables and figures

```bash
# Table 1: sample distribution by size group
python dataset_stats.py

# Per-size-group Recall (run for every model)
python evaluate_size.py --weights runs/yolov8s_baseline/weights/best.pt \
                        --out results/yolov8s_baseline.json
python evaluate_size.py --weights runs/yolov8s_adaptive/weights/best.pt \
                        --out results/yolov8s_adaptive.json
python evaluate_size.py --weights runs/yolo11s_baseline/weights/best.pt \
                        --out results/yolo11s_baseline.json
python evaluate_size.py --weights runs/yolo11s_adaptive/weights/best.pt \
                        --out results/yolo11s_adaptive.json

# Tables 2 + 3 (CSV in results/)
python compare_runs.py \
    --runs   yolov8s_baseline yolov8s_adaptive yolo11s_baseline yolo11s_adaptive \
    --labels "YOLOv8s baseline" "YOLOv8s + adaptive loss" \
             "YOLOv11s baseline" "YOLOv11s + adaptive loss"

# Article figures (figures/)
python figures.py \
    --runs   yolov8s_baseline yolov8s_adaptive yolo11s_baseline yolo11s_adaptive \
    --labels "YOLOv8s baseline" "YOLOv8s + adaptive" \
             "YOLOv11s baseline" "YOLOv11s + adaptive" \
    --n-examples 6
```

## Outputs

```
runs/
  <run_name>/
    weights/best.pt         -- final model
    results.csv             -- per-epoch losses and metrics
    results.png             -- ultralytics default training curves
    PR_curve.png            -- precision-recall curve
    confusion_matrix.png

results/
  table1_distribution.csv     -- sample distribution by size group
  table2_overall.csv          -- P, R, F1, mAP per model
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

## File layout

```
adaptive_loss.py        -- AdaptiveDetectionLoss + enable_adaptive_loss()
train.py                -- training entry point (--adaptive toggle)
predict.py              -- inference on image / video / webcam
download_dataset.py     -- dataset fetch + split
dataset_stats.py        -- Table 1 (size distribution)
evaluate_size.py        -- per-size Recall + localization error
compare_runs.py         -- aggregate Tables 2 and 3
figures.py              -- publication figures
requirements.txt
```

## Notes for Colab

Training on CPU is impractical. For Colab use `Runtime → Change runtime
type → T4 GPU`, clone this repo, then run the same commands.
A T4 finishes one 80-epoch run in ~1.5–2 hours.
