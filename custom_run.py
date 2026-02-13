from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/11/yolov11-SCM.yaml')

result = model.info(verbose=True)

print(result)