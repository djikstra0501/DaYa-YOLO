from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/11/yolov11-color.yaml')
# model = YOLO('YOLO-DP.pt')

# result = model.load("yolo11n.pt")
# result = (model.ckpt["train_args"])
result = model.info(verbose=True)

print(result)