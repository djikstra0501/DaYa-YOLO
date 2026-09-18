import ultralytics
from ultralytics import YOLO
from pathlib import Path

HOME_DIR = Path.home()
DATA_DIR = HOME_DIR / "ultralytics" / "data" / "data.yaml"

model = YOLO("dy_lab.engine")
result = model.val(data=DATA_DIR, split="test", conf=0.01, iou=0.25)