# Drone Detection — YOLOv11

Детекція дронів за допомогою YOLOv11. Датасет завантажується автоматично з Roboflow.

## Встановлення

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Тренування

```bash
python train.py
```

Параметри можна змінити у `train.py` (MODEL_SIZE, EPOCHS, BATCH тощо).

## Інференс

```bash
# Тестові зображення
python predict.py

# Своє зображення або відео
python predict.py --source path/to/image.jpg
python predict.py --source path/to/video.mp4

# Веб-камера
python predict.py --source 0 --show
```

## Структура проекту

```
drone-yolo11/
├── train.py              # тренування
├── predict.py            # інференс
├── download_dataset.py   # завантаження та розбивка датасету
├── requirements.txt
└── README.md
```

## Результати

Після тренування:
- Модель: `drone_detection/yolo11s_run1/weights/best.pt`
- Графіки: `drone_detection/yolo11s_run1/results.png`
