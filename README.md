# Drone Detection — YOLOv11

YOLOv11-based drone detection trained on a Roboflow dataset. Dataset is downloaded and split automatically.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

Training parameters can be adjusted in `train.py` (MODEL_SIZE, EPOCHS, BATCH, etc.).

## Inference

```bash
# Test images
python predict.py

# Custom image or video
python predict.py --source path/to/image.jpg
python predict.py --source path/to/video.mp4

# Webcam
python predict.py --source 0 --show
```

## Project structure

```
drone-yolo11/
├── train.py              # training script
├── predict.py            # inference script
├── download_dataset.py   # dataset download and split
├── requirements.txt
└── README.md
```

## Results

After training:
- Best model: `drone_detection/yolo11s_run1/weights/best.pt`
- Plots: `drone_detection/yolo11s_run1/results.png`
