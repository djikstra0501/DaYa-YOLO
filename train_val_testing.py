from ultralytics import YOLO

# Load your custom model config
model = YOLO("ultralytics/cfg/models/v8/yolov8n-MCBAM.yaml")

# --- TRAIN TEST ---
try:
    print("\n[TEST] Starting tiny training test...")
    model.train(
        data="coco8.yaml",   # small built-in dataset
        epochs=1,            # only 1 epoch (quick test)
        imgsz=256,           # smaller image size
        device="cpu",        # keep it light
        save=False,          # ⛔ don't save weights
        save_period=-1,      # ⛔ never save during training
        project=None,        # ⛔ no folder creation
        name=None,           # ⛔ no run name
        exist_ok=True,
        plots=False    
    )
    print("[TEST] Train OK!")
except Exception as e:
    print("[ERROR] Training failed:")
    print(e)

# --- VAL TEST ---
try:
    print("\n[TEST] Starting validation test...")
    metrics = model.val(
        data="coco8.yaml",
        imgsz=256,
        device="cpu",
        save=False,          # ⛔ don't save results
        plots=False          # ⛔ no plots
    )
    print("[TEST] Val OK!")
    print(metrics)
except Exception as e:
    print("[ERROR] Validation failed:")
    print(e)
