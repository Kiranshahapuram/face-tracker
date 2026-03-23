"""
Face detection module.
Runs YOLOv8 face detector model for finding face/head bounding boxes.
"""

from ultralytics import YOLO
import torch
import cv2
import numpy as np
import logging

class FaceDetector:
    def __init__(self, config):
        self.config = config
        logging.info("Loading YOLOv8 model for face detection...")
        self.model = YOLO(self.config.detection.yolo_model_path)
        torch.set_num_threads(self.config.detection.yolo_num_threads)
        if not self.config.system.use_gpu:
            self.model.to('cpu')
        logging.info("YOLOv8 model loaded.")

    def _is_margin_violated(self, bbox: list, frame_shape: tuple, margin: int) -> bool:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        if cx < margin or cx > (w - margin) or cy < margin or cy > (h - margin):
            return True
        return False

    def crop_face(self, frame: np.ndarray, bbox: list, padding: int = 40) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        # Extra context: expand by 30% of bbox size on each side
        pad_x = int(bw * 0.3) + padding
        pad_y = int(bh * 0.3) + padding
        nx1 = max(0, int(x1) - pad_x)
        ny1 = max(0, int(y1) - pad_y)
        nx2 = min(w, int(x2) + pad_x)
        ny2 = min(h, int(y2) + pad_y)
        return frame[ny1:ny2, nx1:nx2]

    def detect(self, frame: np.ndarray, imgsz: int = 0) -> list[dict]:
        input_size = imgsz if imgsz > 0 else self.config.detection.yolo_input_size
        try:
            results = self.model(
                frame, 
                imgsz=input_size, 
                verbose=False,
                max_det=100  # Support many people in crowd
            )
        except Exception as e:
            logging.error(f"YOLO detection exception: {e}")
            return []

        if len(results) == 0:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        bbs = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        # logging.debug(f"YOLO raw detections: {len(bbs)} boxes found in frame")

        ret = []
        min_size = self.config.detection.min_face_size_px
        margin = self.config.video.roi_margin_px
        h, w = frame.shape[:2]

        for i in range(len(bbs)):
            conf = float(confs[i])
            bbox = bbs[i].tolist()
            
            if conf < self.config.detection.yolo_confidence:
                continue

            if self._is_margin_violated(bbox, frame.shape, margin):
                logging.warning(f"Detection at {bbox} discarded due to ROI margin.")
                continue
            
            crop = self.crop_face(frame, bbox)
            if crop.shape[0] < min_size or crop.shape[1] < min_size:
                # Log filtered out tiny faces as WARNING since user is missing detections
                logging.warning(f"Detection at {bbox} filtered: size {crop.shape[1]}x{crop.shape[0]} < {min_size}")
                continue

            ret.append({
                "bbox": bbox,
                "confidence": conf,
                "face_crop": crop
            })

        if len(ret) > 0:
            logging.info(f"Frame Processed: Found {len(ret)} valid detections (raw YOLO had {len(bbs)}).")
        elif len(bbs) > 0:
            logging.debug(f"Frame Processed: All {len(bbs)} raw detections were filtered out.")
        return ret
