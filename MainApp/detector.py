import os
import cv2
import time
import torch
import numpy as np
import random
from ultralytics import YOLO


model_path = os.path.join("model", "model4.pt")
video_path = os.path.join("video", "videotest2.mp4")
names = ['Helmet White', 'Helmet Yelow', 'Helmet Blue', 'Helmet Red', 'Helmet Orange', 'Jacket']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model và video
model = YOLO(model_path).to(device)
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        id +=1
        results = model.predict(source=frame, save=False, conf=0.4, verbose = False)
        detections = []
        detections_sort = []
        
        # Detection

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().item()
                cls = int(box.cls[0])

                if conf > 0.5:
                    detections_sort.append([x1, y1, x2, y2, conf, cls])

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    label = f"{names[cls]} {int(conf*100)}%"
                    cv2.putText(frame, label, (int(x1), int(y1)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow(f'YOLO Object Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

