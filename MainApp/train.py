import numpy as np
from ultralytics import YOLO

classes = ['head', 'helmet']

model = YOLO("model/yolo11s.pt")

model.train(
    data='data.yaml',  
    epochs=20,         
    imgsz=640,         
    batch=2,          
    device=0          
)

metrics = model.val(data='data.yaml', split='test')
print(metrics)