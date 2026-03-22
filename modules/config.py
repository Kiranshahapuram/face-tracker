"""
Config loader module.
Loads, validates, and exposes configurations from config.json.
"""

import json
from dataclasses import dataclass
import logging
from typing import Any

@dataclass
class VideoConfig:
    source: str
    frame_skip_interval: int
    rtsp_reconnect_delay_s: int
    rtsp_max_retries: int
    roi_margin_px: int

@dataclass
class DetectionConfig:
    yolo_model_path: str
    yolo_confidence: float
    yolo_input_size: int
    yolo_num_threads: int
    min_face_size_px: int

@dataclass
class QualityGateConfig:
    min_blur_score: float
    max_head_angle_deg: int

@dataclass
class ReidConfig:
    insightface_model: str
    similarity_threshold: float
    reid_every_n_frames: int
    embedding_avg_samples: int
    reentry_window_s: int
    reentry_max_buffer: int

@dataclass
class TrackingConfig:
    tracker: str
    max_track_age_frames: int
    min_entry_frames: int
    n_init: int
    entry_line_y: int
    exit_line_y: int

@dataclass
class DatabaseConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str
    pool_min: int
    pool_max: int

@dataclass
class SystemConfig:
    use_gpu: bool
    log_dir: str
    frame_queue_size: int

class Config:
    def __init__(self, path: str = "config.json"):
        with open(path, 'r') as f:
            raw_data = json.load(f)
            
        required_keys = ["video", "detection", "quality_gate", "reid", "tracking", "database", "system"]
        for key in required_keys:
            if key not in raw_data:
                raise ValueError(f"Missing required config section: {key}")

        self.video = VideoConfig(**raw_data["video"])
        self.detection = DetectionConfig(**raw_data["detection"])
        self.quality_gate = QualityGateConfig(**raw_data["quality_gate"])
        self.reid = ReidConfig(**raw_data["reid"])
        self.tracking = TrackingConfig(**raw_data["tracking"])
        self.database = DatabaseConfig(**raw_data["database"])
        self.system = SystemConfig(**raw_data["system"])
        
        logging.basicConfig(level=logging.INFO)
        logging.info("Config loaded successfully.")
