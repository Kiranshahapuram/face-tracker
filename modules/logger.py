"""
Logger module.
Sets up Python logging for all event types and creates necessary log directories.
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

class FaceTrackerLogger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.logger = logging.getLogger("FaceTrackerLog")
        self.logger.setLevel(logging.INFO)
        
        # Format required by spec
        formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
        
        file_path = os.path.join(self.log_dir, "events.log")
        file_handler = RotatingFileHandler(file_path, maxBytes=50*1024*1024, backupCount=5)
        file_handler.setFormatter(formatter)
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(stream_handler)
            
        # create entries and exits per day
        today = datetime.now().strftime("%Y-%m-%d")
        os.makedirs(os.path.join(self.log_dir, "entries", today), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "exits", today), exist_ok=True)

    def log_entry(self, face_id: str, track_id: int, frame_number: int, image_path: str):
        self.logger.info(f"[ENTRY]               face_id={face_id} track_id={track_id} frame={frame_number} image={image_path}")

    def log_exit(self, face_id: str, track_id: int, frame_number: int, image_path: str):
        self.logger.info(f"[EXIT]                face_id={face_id} track_id={track_id} frame={frame_number} image={image_path}")

    def log_tracking(self, face_id: str, track_id: int, frame_number: int):
        self.logger.info(f"[TRACKING]            face_id={face_id} track_id={track_id} frame={frame_number}")

    def log_recognition(self, face_id: str, track_id: int, similarity: float, frame_number: int):
        self.logger.info(f"[RECOGNITION]         face_id={face_id} track_id={track_id} similarity={similarity:.2f} frame={frame_number}")

    def log_embedding_generated(self, track_id: int, frame_number: int, quality_passed: str):
        self.logger.info(f"[EMBEDDING_GENERATED] track_id={track_id} dim=512 quality={quality_passed} frame={frame_number}")

    def log_face_registered(self, face_id: str, track_id: int, unique_count: int, frame_number: int):
        self.logger.info(f"[FACE_REGISTERED]     face_id={face_id} track_id={track_id} unique_count={unique_count} frame={frame_number}")
