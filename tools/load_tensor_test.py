"""
Check pretrained weight transfer for DaYa-YOLO vs Base.

Prints the "Transferred X/Y items" line that Ultralytics emits during load,
plus a breakdown of which layers actually received weights. This is the number
in dispute (14/532 vs 499/499).
"""

from ultralytics import YOLO

# ============================================================
# CONFIG — point these at your actual files
# ============================================================
CONFIGS = {
    "Base (YOLO11n)": "ultralytics/cfg/models/11/yolo11n.yaml",
    "DaYa-YOLO":       "ultralytics/cfg/models/11/yolo11-c-aux.yaml",
    "RGB Identity":    "ultralytics/cfg/models/11/yolo11-r-aux.yaml",
    "EMA":       "ultralytics/cfg/models/11/yolov11-EMA-REF.yaml",
    "LSKA":       "ultralytics/cfg/models/11/yolov11-LSKA-REF.yaml",
    "BRA":       "ultralytics/cfg/models/11/yolov11-BRA-REF.yaml",
}
PRETRAINED = "yolo11n.pt"   # the COCO checkpoint used for all runs


for name, cfg in CONFIGS.items():
    print("\n" + "=" * 70)
    print(f"{name}  ({cfg})")
    print("=" * 70)

    # Build from YAML (random init), then load pretrained.
    # This is the same path model.train(pretrained=...) takes, so the
    # "Transferred" line printed here is the one your training runs saw.
    model = YOLO(cfg)

    # Confirm which loader branch fires — your load() prints one of:
    #   "[Weight Router] DaYa-YOLO detected. Using Decoupled Spectral Loader..."
    #   "Using standard custom loader due to architecture mismatch."
    #   "Same architecture, skipping custom loader."
    model.load(PRETRAINED)

    # Where does SpectralFeatureEncoder actually sit?
    # is_daya_architecture() returns i > 0, so index 0 SKIPS the DaYa loader.
    for i, layer in enumerate(model.model.model):
        if layer.__class__.__name__ == "SpectralFeatureEncoder":
            print(f"  SpectralFeatureEncoder at index {i}"
                  f"  -> DaYa loader {'ENGAGED' if i > 0 else 'SKIPPED (index 0)'}")

    # Sanity check: how many layers are still at their init values?
    total = len(model.model.state_dict())
    print(f"  Total tensors in model: {total}")