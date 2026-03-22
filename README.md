# Intelligent Face Tracker

## Setup Instructions

1.  **Environment**: Ensure Python 3.11+ is installed.
2.  **Dependencies**: Install required packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Database**:
    -   Ensure Docker Desktop is running.
    -   Start PostgreSQL with pgvector:
        ```bash
        docker compose up -d
        ```
    -   The database is accessible on port `6432`.
4.  **Model Weights**:
    -   The YOLOv8 face model `yolov8n-face.pt` is downloaded automatically by `setup.py` or manually from a verified GitHub release.
    -   InsightFace models (`buffalo_l`) are downloaded on the first run.
5.  **Initialization**:
    ```bash
    python setup.py
    ```

## How to Run

```bash
# Run with default config (source: sample.mp4)
python main.py --config config.json

# Override video source:
python main.py --config config.json --source path/to/video.mp4

# RTSP stream (interview mode):
python main.py --config config.json --source rtsp://camera_ip/stream
```

## Architecture

The system utilizes a **4-thread asynchronous pipeline**:
-   **T1 (Capture)**: Decodes frames and manages RTSP reconnection.
-   **T2 (Tracker)**: Handles YOLOv8 detection and ByteTrack association.
-   **T3 (Re-ID)**: Performs InsightFace embedding extraction and global identity resolution.
-   **T4 (Async I/O)**: Manages database writes and image saves without blocking the real-time processing threads.

## AI Planning Document

-   **ByteTrack vs DeepSort**: ByteTrack was chosen for its superior performance in high-density crowds and lower computational overhead, which is critical for CPU-based inference.
-   **Buffalo_L vs Buffalo_S**: Used the larger `buffalo_l` model for higher precision in ArcFace embeddings (512-dim) to ensure accurate cross-visit identification.
-   **Two-Layer Re-ID**: Implements an active track binding for within-visit consistency and a 120-second re-entry buffer to handle temporary occlusions or rapid exits and re-entries.
-   **Database Choice**: PostgreSQL with `pgvector` allows for highly efficient vector similarity searches (`ORDER BY embedding <=> %s::vector`) compared to standard BLOB storage.

## Sample config.json

```json
{
  "video": {
    "source": "sample.mp4",
    "frame_skip_interval": 5,
    "rtsp_reconnect_delay_s": 5,
    "rtsp_max_retries": 10,
    "roi_margin_px": 40
  },
  "detection": {
    "yolo_model_path": "yolov8n-face.pt",
    "yolo_confidence": 0.5,
    "yolo_input_size": 640,
    "yolo_num_threads": 4,
    "min_face_size_px": 50
  },
  "quality_gate": {
    "min_blur_score": 80.0,
    "max_head_angle_deg": 65
  },
  "reid": {
    "insightface_model": "buffalo_l",
    "similarity_threshold": 0.45,
    "reid_every_n_frames": 25,
    "embedding_avg_samples": 3,
    "reentry_window_s": 120,
    "reentry_max_buffer": 200
  },
  "tracking": {
    "tracker": "bytetrack",
    "max_track_age_frames": 30,
    "min_entry_frames": 3,
    "n_init": 2,
    "entry_line_y": 60,
    "exit_line_y": 660
  },
  "database": {
    "host": "localhost",
    "port": 6432,
    "dbname": "face_tracker",
    "user": "postgres",
    "password": "",
    "pool_min": 2,
    "pool_max": 5
  },
  "system": {
    "use_gpu": false,
    "log_dir": "logs/",
    "frame_queue_size": 4
  }
}
```

## Assumptions Made
- Camera is overhead/ceiling mounted (40–80° angle) — thresholds tuned accordingly.
- CPU-only inference — throughput ~15 FPS effective, acceptable for visitor counting.
- Masked and niqab-covered faces are counted via body detection but cannot be Re-ID'd on re-entry.
- Groups walking tightly together may be undercounted due to merged bounding boxes.
- Zone lines placed near frame edges where people are most separated.

## Known Limitations

| Limitation | Impact | Mitigation |
| :--- | :--- | :--- |
| Harsh Backlighting | Face detection failure | Contrast normalization in preprocessing |
| Low Frame Rate | Lower tracking stability | Reduced ByteTrack threshold and increased buffer |
| Network Latency | RTSP lag | Frame dropping queuing strategy |

## Compute Load Estimation

| Module | Thread | CPU Load | Time per call |
| :--- | :--- | :--- | :--- |
| YOLOv8 | T2 | High | ~40ms |
| ByteTrack | T2 | Low | ~5ms |
| InsightFace | T3 | High | ~150ms |
| DB Writes | T4 | Low | ~10ms |

## Sample Output

```
2026-03-21 11:31:01,885 | INFO     | [EMBEDDING_GENERATED] track_id=1 dim=512 quality=passed frame=80
2026-03-21 11:31:02,885 | INFO     | [RECOGNITION]         face_id=e6459341-3b8c-4d80-87a3-e4d56789abcd track_id=1 similarity=0.00 frame=90
2026-03-21 11:31:03,885 | INFO     | [FACE_REGISTERED]     face_id=e6459341-3b8c-4d80-87a3-e4d56789abcd track_id=1 unique_count=1 frame=95
2026-03-21 11:31:04,885 | INFO     | [ENTRY]               face_id=e6459341-3b8c-4d80-87a3-e4d56789abcd track_id=1 frame=100 image=logs/entries/2026-03-21/e645...jpg
2026-03-21 11:31:09,885 | INFO     | [TRACKING]            face_id=e6459341-3b8c-4d80-87a3-e4d56789abcd track_id=1 frame=110
```

## Demo Video

[Link to Demo Video](https://youtube.com/example_link)

This project is a part of a hackathon run by https://katomaran.com
