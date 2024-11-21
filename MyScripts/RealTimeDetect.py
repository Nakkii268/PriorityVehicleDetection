import os
import cv2
from ultralytics import YOLO

# Load your custom YOLO model
model_path = os.path.join('.', 'runs', 'detect', 'train16', 'weights', 'last.pt')

model = YOLO(model_path)  # load a custom model

# Load class names (assuming they are the same as coco.names)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Specify the classes to detect
target_classes = ["ambulance", "Police", "firetruck"]

# Open video capture
VIDEOS_DIR = os.path.join('.', 'videos')

video_path = os.path.join(VIDEOS_DIR, 'ambulacetest.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform object detection
    results = model(frame)

    # Process results and draw bounding boxes
    for result in results:
        boxes = result.boxes.xyxy  # Bounding boxes
        scores = result.boxes.conf  # Confidence scores
        classes = result.boxes.cls  # Class IDs

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(cls)]
            if class_name in target_classes:
                color = (0, 255, 255) if class_name == "ambulance" else (255, 0, 0) if class_name == "Police" else (0, 255,0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f'{class_name} {score:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame with detections
    cv2.imshow('YOLOv8 Object Detection', frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
