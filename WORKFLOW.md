# AI Application Development Workflow: Intelligent Face Tracker

This document outlines the systematic workflow followed to build the Intelligent Face Tracker application, including the architectural planning, feature set documentation, and computational load analysis.

---

## 1. Planning the Application

The application was built using a performance-first, multi-threaded architectural approach to ensure that high-latency AI tasks (like recognition) do not block real-time tasks (like video capture).

### Architectural Strategy: The Four-Thread Pipeline
To achieve smooth processing, the backend was split into four specialized concurrent workers:
- **T1: Capture Worker**: Dedicated solely to dequeuing frames from the RTSP/File stream to prevent buffer overflow.
- **T2: Tracking Worker**: Executes the YOLOv8 face detector and ByteTrack tracker at the highest possible framerate.
- **T3: Recognition Worker**: Handles the "heavy lifting" of generating 512-D embeddings using ArcFace. It processes faces asynchronously from the tracker.
- **T4: I/O Worker**: Manages database persistence (Events/Sessions) and stores cropped face images to the disk without interrupting the vision analysis.

### Technical Stack Selection
- **Detection**: YOLOv8n-face (Optimized for speed/accuracy balance).
- **Recognition**: ArcFace (via InsightFace) for high-precision identity consistency.
- **Database**: PostgreSQL + pgvector (Enables native, sub-second vector similarity searches).
- **Communication**: FastAPI for the REST surface + React for the data-dense dashboard.

---

## 2. Feature Documentation

The Intelligent Face Tracker performs the following core operations:

### Multi-Target Face Tracking
- **Capability**: Detects and uniquely tracks multiple individuals simultaneously using the ByteTrack algorithm.
- **Persistence**: Assigns a `Track ID` that follows a person even if they are briefly occluded or move across the frame.

### Deep Search Re-Identification (Re-ID)
- **Capability**: Converts detected faces into mathematical embeddings.
- **Matching**: Automatically queries the database to see if the visitor has been seen in previous sessions, maintaining a "Unique Visitor" vs. "Repeat Visitor" state.

### Automated Entry/Exit Logic
- **Capability**: A state machine monitors the persistence of a track.
- **Logic**: Registers an "Entry" when a face is consistently seen for `N` frames and an "Exit" when the track is lost for more than a set temporal window.

### Live Visualization & HUD
- **Capability**: Renders a real-time debug window showing bounding boxes, track metadata, and global visitor statistics.
- **Optimization**: Uses a dedicated main-thread loop to remain compatible with Windows GUI event handling.

### Analytics Dashboard
- **Capability**: A centralized web interface listing recent visitors, event logs with thumbnails, session-wise statistics, and raw log file downloads.

---

## 3. Compute Load Estimation

The system's performance varies based on whether hardware acceleration (CUDA) is available. Below are the estimated consumptions for a 1080p stream at 30 FPS.

### CPU Consumption (Per 30 FPS Stream)
- **Detection (YOLOv8n)**: ~25-40% utilization on a modern Hexa-core processor. Latency: 30-50ms per frame.
- **Recognition (ArcFace)**: High load. ~15-20% per active face being re-identified. Latency: 150-250ms per face.
- **Pipeline Overheads**: ~5-10% (Frame resizing, queue management, and OpenCV rendering).
- **Total System RAM**: ~1.2GB - 1.8GB (primarily for model weights and frame buffers).

### GPU Consumption (NVIDIA CUDA)
- **Detection (YOLOv8n)**: Very low. ~10-15% Tensor core utilization. Latency: 3-5ms per frame.
- **Recognition (ArcFace)**: Moderate. ~30% utilization during peak registration events. Latency: 8-12ms per face.
- **VRAM Usage**: ~1.5GB (Fixed load for model allocation).
- **Total System Power**: Optimized for real-time performance without hitting thermal limits of mid-range GPUs.

### Database Impact
- **Search Latency**: <5ms for up to 100,000 registered face embeddings using HNSW or Ivfflat indexes in pgvector.
- **Disk I/O**: Intermittent bursts (50-200 KB) during Entry/Exit image storage events.

---

## 4. Development Workflow Phases

1.  **Phase 0 (Foundation)**: Containerized PostgreSQL setup and pgvector schema design.
2.  **Phase 1 (Vision)**: Implementation of the T1+T2 pipeline (Capture -> Detection -> Tracking).
3.  **Phase 2 (Identity)**: Integration of T3 ArcFace and vector matching logic.
4.  **Phase 3 (State)**: Implementation of the Entry/Exit state machine and visit counter persistence.
5.  **Phase 4 (Interface)**: Development of the FastAPI service and the React dashboard.
6.  **Phase 5 (Refinement)**: Final performance tuning, stop-session reliability, and logging enhancements.
