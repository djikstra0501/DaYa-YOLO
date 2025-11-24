from ultralytics import YOLO
model = YOLO('yolov11-eca-sam-s2.pt')

data_stage_2 = "ultralytics/data/general/stage_2/data.yaml"

model.val(data=data_stage_2, split='test', save_json=True, save_txt=True)