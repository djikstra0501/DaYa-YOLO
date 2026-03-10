import os
import json
from pathlib import Path

from sahi.predict import get_sliced_prediction
from sahi.models.ultralytics import UltralyticsDetectionModel

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


MODEL_PATH = "yolov11-eca-sam-s3.pt"

IMAGE_DIR = os.path.join(data_split_info, "valid")
GT_JSON = os.path.join(data_split_info, "valid", "_annotations.coco.json")

PRED_JSON = "sahi_predictions.json"

SLICE_SIZE = 320
OVERLAP = 0.25


# load model
detection_model = UltralyticsDetectionModel(
    model_path=MODEL_PATH,
    confidence_threshold=0.25,
    device="cuda"
)

detection_model.load_model()

predictions = []

cocoGt = COCO(GT_JSON)

for img in cocoGt.dataset["images"]:

    img_path = os.path.join(IMAGE_DIR, img["file_name"])

    result = get_sliced_prediction(
        img_path,
        detection_model,
        slice_height=SLICE_SIZE,
        slice_width=SLICE_SIZE,
        overlap_height_ratio=OVERLAP,
        overlap_width_ratio=OVERLAP,
    )

    for obj in result.object_prediction_list:

        bbox = obj.bbox.to_xywh()

        predictions.append({
            "image_id": img["id"],
            "category_id": obj.category.id,
            "bbox": bbox,
            "score": obj.score.value
        })


with open(PRED_JSON, "w") as f:
    json.dump(predictions, f)

# evaluate
cocoDt = cocoGt.loadRes(PRED_JSON)

print("Model classes:", detection_model.model.names)
print("COCO classes:", cocoGt.dataset["categories"])

cocoEval = COCOeval(cocoGt, cocoDt, "bbox")

cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()