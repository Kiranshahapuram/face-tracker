"""
Tracking wrapper module.
Uses ByteTrack for handling and updating object tracks across frames.
"""

from boxmot import ByteTrack
import numpy as np
import logging

class FaceTracker:
    def __init__(self, config):
        self.config = config
        # match_thresh: IoU threshold for matching detections to existing tracks.
        #   0.8 is way too strict for CCTV — walking people's bboxes shift 10-20px/frame,
        #   which drops IoU below 0.8 for small faces (~40px). 0.45 is standard for MOT.
        # frame_rate: must match actual video FPS for correct Kalman filter predictions.
        self.tracker = ByteTrack(
            track_thresh=self.config.detection.yolo_confidence,
            track_buffer=self.config.tracking.max_track_age_frames,
            match_thresh=0.9,  # Distance threshold (1 - IoU). 0.9 means we tolerate IoU down to 0.1
            frame_rate=25
        )
        self.active_track_ids = set()


    def update(self, detections: list[dict], frame: np.ndarray) -> list[dict]:
        if not detections:
            tracks = self.tracker.update(np.empty((0, 6)), frame)
        else:
            dets = []
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                conf = d["confidence"]
                # BYTETracker expects [x1, y1, x2, y2, conf, cls] 
                dets.append([x1, y1, x2, y2, conf, 0])
            
            dets = np.array(dets)
            tracks = self.tracker.update(dets, frame)

        # tracks format: [x1, y1, x2, y2, id, conf, cls, ind]
        confirmed_tracks = []
        new_active = set()
        for t in tracks:
            track_id = int(t[4])
            x1, y1, x2, y2 = t[:4]
            
            confirmed_tracks.append({
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2],
                "is_confirmed": True
            })
            new_active.add(track_id)
                
        self.active_track_ids = new_active
        
        if len(confirmed_tracks) > 0:
            logging.debug(f"Tracker: {len(detections)} detections -> {len(confirmed_tracks)} active tracks (IDs: {new_active})")
        
        return confirmed_tracks

    def get_active_track_ids(self) -> set[int]:
        return self.active_track_ids
