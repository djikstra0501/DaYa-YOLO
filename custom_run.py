from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/11/yolov11-eca-sam.yaml')

result = model.load("yolo11n.pt")
# result = model.info(verbose=True)

print(result)