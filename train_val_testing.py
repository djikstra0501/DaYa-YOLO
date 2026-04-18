from ultralytics import YOLO
import traceback

# Load your custom model config
model = YOLO("ultralytics/cfg/models/11/yolov11-color.yaml")

# --- TRAIN TEST ---
try:
    print("\n[TEST] Starting tiny training test...")
    model.train(
        data="coco8.yaml",   # small built-in dataset
        epochs=10,           # train a bit longer so head can learn
        imgsz=256,           # smaller image size
        device="cpu",        # keep it light
        pretrained="yolo11n.pt",  # warm start from base weights
        save=False,          # don't save weights
        save_period=-1,      # never save during training
        project=None,        # no folder creation
        name=None,           # no run name
        exist_ok=True,
        plots=False,
        freeze_epochs=5,        # freeze backbone for first 5 epochs
        freeze_layers="1-10",  # specify layers to freeze
    )
    print("[TEST] Train OK!")
except Exception as e:
    traceback.print_exc()
    print("[ERROR] Training failed:")
    print(e)

# --- VAL TEST ---
try:
    print("\n[TEST] Starting validation test...")
    metrics = model.val(
        data="coco8.yaml",
        imgsz=256,
        device="cpu",
        save=False,          # don't save results
        plots=False,         # no plots
    )
    print("[TEST] Val OK!")
    print(metrics)
except Exception as e:
    traceback.print_exc()
    print("[ERROR] Validation failed:")
    print(e)
