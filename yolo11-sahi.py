from sahi import AutoDetectionModel
from sahi.predict import predict

predict(
    model_type="ultralytics",
    model_path="best.pt",
    source="path/to/val/images",
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    export_pickle=True,
    export_json=True
)