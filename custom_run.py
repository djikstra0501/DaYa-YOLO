from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/11/yolov11-LSKA-REF.yaml')
# model = YOLO('yolo11n.pt')

# result = model.load("yolo11n.pt")
# result = (model.ckpt["train_args"])
result = model.info(verbose=True)

print(result)