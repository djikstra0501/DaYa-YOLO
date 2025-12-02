from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/v8/yolov8n-MCBAM.yaml')

result = model.info(verbose=True)

print(result)