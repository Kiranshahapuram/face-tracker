"""
Diagnostic script - writes results to JSON to avoid terminal rendering issues.
"""
import cv2
import numpy as np
from ultralytics import YOLO
import json

MODEL_PATH = "yolov8n-face.pt"
VIDEO_PATH = "uploads\\record_20250620_183903.mp4"

results_data = {}

def test_on_image(model, img_path):
    frame = cv2.imread(img_path)
    if frame is None:
        return {"error": "could not read"}
    h, w = frame.shape[:2]
    img_result = {"size": f"{w}x{h}", "tests": {}}
    
    for imgsz in [640, 1280]:
        res = model(frame, imgsz=imgsz, verbose=False, max_det=50)
        if len(res) == 0 or res[0].boxes is None or len(res[0].boxes) == 0:
            img_result["tests"][str(imgsz)] = {"count": 0}
            continue
        
        boxes = res[0].boxes
        confs = boxes.conf.cpu().numpy()
        bbs = boxes.xyxy.cpu().numpy()
        
        dets = []
        for i in range(len(bbs)):
            x1, y1, x2, y2 = bbs[i].tolist()
            c = float(confs[i])
            bw = x2 - x1
            bh = y2 - y1
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            at_edge = cx < 50 or cx > (w - 50) or cy < 50 or cy > (h - 50)
            dets.append({
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
                "size": f"{round(bw)}x{round(bh)}",
                "conf": round(c, 3),
                "at_edge": at_edge
            })
        
        img_result["tests"][str(imgsz)] = {
            "count": len(dets),
            "count_conf_015": int(np.sum(confs >= 0.15)),
            "detections": dets
        }
    
    return img_result

def test_on_video(model, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "could not open"}
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    video_result = {
        "resolution": f"{w}x{h}",
        "fps": fps,
        "total_frames": total,
        "frames": {}
    }
    
    test_frames = [50, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 1200, 1400]
    for fn in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            video_result["frames"][str(fn)] = {"error": "read failed"}
            continue
        
        res = model(frame, imgsz=1280, verbose=False, max_det=50)
        if len(res) > 0 and res[0].boxes is not None and len(res[0].boxes) > 0:
            boxes = res[0].boxes
            confs = boxes.conf.cpu().numpy()
            bbs = boxes.xyxy.cpu().numpy()
            
            dets = []
            for i in range(len(bbs)):
                x1, y1, x2, y2 = bbs[i].tolist()
                c = float(confs[i])
                bw = x2 - x1
                bh = y2 - y1
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                at_edge = cx < 50 or cx > (w - 50) or cy < 50 or cy > (h - 50)
                dets.append({
                    "bbox": [round(x1), round(y1), round(x2), round(y2)],
                    "size": f"{round(bw)}x{round(bh)}",
                    "center": [round(cx), round(cy)],
                    "conf": round(c, 3),
                    "at_edge": at_edge
                })
            
            video_result["frames"][str(fn)] = {
                "count": len(dets),
                "count_conf_015": int(np.sum(confs >= 0.15)),
                "detections": dets
            }
        else:
            video_result["frames"][str(fn)] = {"count": 0}
    
    cap.release()
    return video_result

if __name__ == "__main__":
    model = YOLO(MODEL_PATH)
    
    results_data["model"] = {
        "path": MODEL_PATH,
        "task": str(model.task),
        "names": {str(k): v for k, v in model.names.items()} if hasattr(model, 'names') else {}
    }
    
    import glob
    debug_frames = sorted(glob.glob("debug_frame_*.jpg"))
    for df in debug_frames:
        results_data[df] = test_on_image(model, df)
    
    results_data["video"] = test_on_video(model, VIDEO_PATH)
    
    with open("diagnose_results.json", "w") as f:
        json.dump(results_data, f, indent=2)
    
    print("DONE - results in diagnose_results.json")
