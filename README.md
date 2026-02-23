# Traffic Density Estimation using YOLOv8

This project estimates traffic density by detecting and counting vehicles in video frames using a pre-trained YOLOv8 object detection model. The system processes video input, detects vehicles in each frame, and calculates traffic density in real time.

## Objective
To analyze traffic flow by counting vehicles within designated regions in video footage using deep learning-based object detection.

## Tech Stack
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy

## Model Used
YOLOv8 pre-trained on the COCO dataset for vehicle detection:
- Car
- Bus
- Truck
- Motorcycle

## Features
- Real-time vehicle detection
- Vehicle counting per frame
- Traffic density estimation
- Bounding box visualization
- Video input and output support

## How It Works
1. Input traffic video is read frame by frame.
2. YOLOv8 detects vehicles using COCO-trained weights.
3. Vehicles are counted within each frame.
4. Traffic density is displayed on screen.
5. Output video with detections is saved.

## How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Run detection:
python yolov8_vehicle_count.py

3. Output video will be saved in the output folder.

## Applications
- Smart city traffic monitoring
- Traffic flow analysis
- Urban planning
- Surveillance systems

## Author
Sameek Bhoir

