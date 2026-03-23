from ultralytics import YOLO
import cv2

model = YOLO("yolov8n-face.pt")
cap = cv2.VideoCapture("uploads/record_20250620_183903.mp4")

for frame_num in [1, 5, 10, 15, 20, 25, 30]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        print(f"Frame {frame_num}: Could not read")
        continue
    results = model(frame, imgsz=1280, verbose=False)
    boxes = results[0].boxes
    print(f"Frame {frame_num}: {len(boxes)} detections")
    for b in boxes:
        print(f"  conf={float(b.conf[0]):.3f} bbox={b.xyxy[0].tolist()}")

cap.release()
