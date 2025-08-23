from ultralytics import YOLO
import cv2

model = YOLO("model/model3.pt")

image_path = "test01.jpg"  
image = cv2.imread(image_path)

results = model.predict(source=image, save=False, conf=0.5)

for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

cv2.imwrite("predicted_image.jpg", image)
cv2.imshow("Prediction", image)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break

cv2.destroyAllWindows()
