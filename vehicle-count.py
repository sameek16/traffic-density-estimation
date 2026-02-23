import cv2
from ultralytics import YOLO

# Load YOLOv8 model (auto downloads)
model = YOLO("yolov8n.pt")

# Vehicle classes from COCO
vehicle_classes = [2, 3, 5, 7]  
# 2=car, 3=motorcycle, 5=bus, 7=truck

# Input video
cap = cv2.VideoCapture("input_videos/traffic.mp4")

# Output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output/output_video.mp4', fourcc, 20.0,
                      (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    vehicle_count = 0

    for box in results.boxes:
        cls = int(box.cls[0])

        if cls in vehicle_classes:
            vehicle_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]

            label = f"Vehicle {conf:.2f}"

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Display vehicle count
    cv2.putText(frame, f"Vehicle Count: {vehicle_count}",
                (20,50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,0,255), 3)

    out.write(frame)
    cv2.imshow("Traffic Density Estimation", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
