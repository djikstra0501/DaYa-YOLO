from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/v8/yolov8n-DP.yaml')
# model = YOLO('YOLO-DP.pt')

# result = model.load("yolo11n.pt")
# result = (model.ckpt["train_args"])
result = model.info(verbose=True)

print(result)