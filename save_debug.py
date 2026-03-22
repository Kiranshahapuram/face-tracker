import cv2
import os

cap = cv2.VideoCapture('uploads/record_20250620_183903.mp4')
f_count = 0
while True:
    ret, f = cap.read()
    if not ret:
        break
    f_count += 1
    if f_count % 300 == 0:
        cv2.imwrite(f'debug_frame_{f_count}.jpg', f)
cap.release()
print(f"Total frames: {f_count}, saved snapshots.")
