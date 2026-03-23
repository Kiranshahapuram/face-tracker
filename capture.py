"""
Pipeline runner module — FIXED VERSION
All four bugs resolved:
1. _crop_from_frame crops directly from frame, no IoU matching
2. T3 exits based on t2.is_alive(), not stop_event
3. T4 exits based on t2+t3 alive, not stop_event
4. No t3.join() in T2 drain (was causing deadlock)
"""

import threading
import queue
import time
import numpy as np
import cv2
import os
import logging
from datetime import datetime

from modules.config import Config
from modules.logger import FaceTrackerLogger
from modules.database import Database
from modules.detector import FaceDetector
from modules.embedder import FaceEmbedder
from modules.recognizer import FaceRecognizer
from modules.tracker import FaceTracker
from modules.state_machine import FaceStateMachine


class Pipeline:
    def __init__(self, config: Config, db: Database, logger: FaceTrackerLogger, session_id: str):
        self.config = config
        self.db = db
        self.logger = logger
        self.session_id = session_id

        self.detector = FaceDetector(config)
        self.embedder = FaceEmbedder(config)
        self.recognizer = FaceRecognizer(config)
        self.tracker = FaceTracker(config)
        self.state_machine = FaceStateMachine(config)

        self.frame_queue = queue.Queue(maxsize=config.system.frame_queue_size)
        self.reid_queue = queue.Queue()
        self.reid_result_queue = queue.Queue()
        self.io_queue = queue.Queue()
        self.vis_queue = queue.Queue(maxsize=2)  # for main-thread visualization
        self._last_frame = None
        self.reid_queued = set()
        self.stop_event = threading.Event()

        self.t1 = threading.Thread(target=self._capture_worker, name="T1-Capture", daemon=True)
        self.t2 = threading.Thread(target=self._tracker_worker, name="T2-Tracker", daemon=True)
        self.t3 = threading.Thread(target=self._reid_worker,    name="T3-ReID",    daemon=True)
        self.t4 = threading.Thread(target=self._io_worker,      name="T4-IO",      daemon=True)

    def start(self):
        logging.info("Starting pipeline threads...")
        self.t4.start(); self.t3.start(); self.t2.start(); self.t1.start()

    def stop(self):
        logging.info("Stopping pipeline...")
        self.stop_event.set()

    def join(self):
        for t in [self.t1, self.t2, self.t3, self.t4]:
            t.join(timeout=10)

    def _render_debug_frame(self, frame: np.ndarray, active_tracks: list,
                             states: dict, unique_count: int, frame_number: int):
        """
        Renders a debug frame with overlays and puts it into vis_queue.
        Called from T2 tracker thread. Does NOT call cv2.imshow (that must
        happen on the main thread on Windows).
        """
        try:
            vis = frame.copy()

            # Scale down for display — original is 2688x1520, too large for screen
            display_w, display_h = 1280, 720
            scale_x = display_w / frame.shape[1]
            scale_y = display_h / frame.shape[0]
            vis = cv2.resize(vis, (display_w, display_h))

            for track in active_tracks:
                x1, y1, x2, y2 = track['bbox']
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                track_id = track['track_id']

                # Look up face_id for this track if available
                face_id = states.get(track_id, {}).get('face_id', None)
                short_id = face_id[:8] if face_id else "pending"

                # Color: yellow if pending Re-ID, green if identified
                color = (0, 255, 0) if face_id else (0, 255, 255)

                # Bounding box
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

                # Label background
                label = f"T:{track_id} {short_id}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.rectangle(vis, (x1, y1 - 16), (x1 + label_size[0] + 4, y1), color, -1)
                cv2.putText(vis, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

            # Top bar — unique visitor count
            cv2.rectangle(vis, (0, 0), (320, 45), (0, 0, 0), -1)
            cv2.putText(vis, f"Unique Visitors: {unique_count}", (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

            # Bottom bar — frame number and active tracks
            cv2.rectangle(vis, (0, display_h - 30), (display_w, display_h), (0, 0, 0), -1)
            cv2.putText(vis, f"Frame: {frame_number}  Active Tracks: {len(active_tracks)}",
                        (10, display_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            # Put into vis_queue for main thread to display (drop if full)
            try:
                self.vis_queue.put_nowait(vis)
            except queue.Full:
                pass  # drop frame, main thread is behind

        except Exception as e:
            logging.warning(f"Visualization render error: {e}")

    def run_visualization_loop(self):
        """
        Must be called from the MAIN thread.
        Polls vis_queue and displays frames with cv2.imshow.
        On Windows, cv2.imshow/waitKey MUST run on the main thread.
        Returns when pipeline stops.
        """
        import os
        stop_file = "stop_flag.txt"
        logging.info("Visualization loop started on main thread.")
        while not self.stop_event.is_set():
            # Check for file-based stop flag
            if os.path.exists(stop_file):
                logging.info("Stop flag detected in viz loop. Stopping pipeline...")
                self.stop()
                try: os.remove(stop_file)
                except: pass
                break

            try:
                vis = self.vis_queue.get(timeout=0.1)
                cv2.imshow("Face Tracker - Live", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return  # user closed viz, pipeline keeps running
            except queue.Empty:
                cv2.waitKey(1)
                continue
        cv2.destroyAllWindows()
        logging.info("Visualization window closed.")

    def _compute_adaptive_imgsz(self, width: int, height: int) -> int:
        """
        Dynamically select YOLO input size based on source resolution.
        Ensures faces are large enough for YOLO to detect regardless of camera.
        
        4K (2160p+)   → 1920  (6x downscale without this would lose small faces)
        HD+ (1400p+)  → 1280  (correct for 1520p mall cameras)
        720p range    → 640   (standard, fast)
        Low res       → 480   (RTSP streams, edge cameras)
        """
        short_side = min(width, height)
        if short_side >= 2000:
            return 1920
        elif short_side >= 1400:
            return 1280
        elif short_side >= 700:
            return 640
        else:
            return 480

    def _capture_worker(self):
        source = self.config.video.source
        is_rtsp = source.startswith("rtsp://")
        cap = cv2.VideoCapture(source)
        frame_number = 0
        retries = 0

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        cfg_imgsz = self.config.detection.yolo_input_size
        if cfg_imgsz == 0:
            self._adaptive_imgsz = self._compute_adaptive_imgsz(int(width), int(height))
            logging.info(f"Using adaptive yolo imgsz: {self._adaptive_imgsz}")
        else:
            self._adaptive_imgsz = cfg_imgsz
            
        self.config.detection.yolo_input_size = self._adaptive_imgsz

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if is_rtsp and retries < self.config.video.rtsp_max_retries:
                    logging.warning(f"RTSP stream lost. Retrying in {self.config.video.rtsp_reconnect_delay_s}s...")
                    time.sleep(self.config.video.rtsp_reconnect_delay_s)
                    cap = cv2.VideoCapture(source)
                    retries += 1
                    continue
                else:
                    logging.info("End of video stream or max retries reached.")
                    break

            frame_number += 1
            if frame_number % self.config.video.frame_skip_interval != 0:
                continue
            if is_rtsp:
                try:
                    self.frame_queue.put((frame, frame_number), block=False)
                except queue.Full:
                    pass  # drop frame for live stream
            else:
                # Wait until T2 has space, but handle T2 crashes
                while True:
                    if self.stop_event.is_set() or not self.t2.is_alive():
                        break
                    try:
                        self.frame_queue.put((frame, frame_number), timeout=1.0)
                        break
                    except queue.Full:
                        pass

        cap.release()
        self.stop_event.set()
        logging.info("T1 capture worker stopped.")

    def _tracker_worker(self):
        reid_interval = self.config.reid.reid_every_n_frames
        self._last_frame = None

        while not self.stop_event.is_set() or not self.frame_queue.empty():
            try:
                frame, frame_number = self.frame_queue.get(timeout=1.0)
                self._last_frame = frame
            except queue.Empty:
                continue

            detections = self.detector.detect(frame, imgsz=self._adaptive_imgsz)
            active_tracks = self.tracker.update(detections, frame)

            # Drain Re-ID results
            while not self.reid_result_queue.empty():
                try:
                    tid, fid, emb, sim, is_new, rfn, crop = self.reid_result_queue.get_nowait()
                    self.reid_queued.discard(tid)
                    if fid is not None:
                        self.state_machine.bind_face_id(tid, fid, emb, is_new, rfn, frame,
                                                        self.db, self.logger, self.session_id, self.io_queue, crop)
                except queue.Empty:
                    break

            self.state_machine.update(active_tracks, frame_number, frame,
                                      self.recognizer, self.embedder, self.db,
                                      self.logger, self.session_id, self.io_queue)

            if getattr(self.config.system, 'show_visualization', False):
                vis_states = {}
                for face_id, state_obj in self.state_machine.states.items():
                    vis_states[state_obj.track_id] = {
                        'face_id': face_id,
                        'state': state_obj.state
                    }
                unique_count = len(self.state_machine._entered_face_ids)
                self._render_debug_frame(frame, active_tracks, vis_states, unique_count, frame_number)

            for track in active_tracks:
                tid = track['track_id']
                if tid in self.state_machine.temp_states:
                    sobj = self.state_machine.temp_states[tid]
                    if sobj.pending_frames >= self.config.tracking.min_entry_frames and tid not in self.reid_queued:
                        crop = self._crop_from_frame(frame, track['bbox'])
                        if crop is not None:
                            self.reid_queued.add(tid)
                            self.reid_queue.put((tid, crop, frame_number, True)) # is_reg=True
                            logging.info(f"T2: queued track {tid} for Re-ID (REG) crop={crop.shape}")
                elif tid in self.state_machine.track_to_face:
                    if frame_number % reid_interval == 0 and tid not in self.reid_queued:
                        face_id = self.state_machine.track_to_face[tid]
                        state_obj = self.state_machine.states.get(face_id)
                        
                        if state_obj and state_obj.last_embedding is not None:
                            crop = self._crop_from_frame(frame, track['bbox'])
                            if crop is not None:
                                self.reid_queued.add(tid)
                                self.reid_queue.put((tid, crop, frame_number, False)) # is_reg=False
                                logging.info(f"T2: Already registered face_id={face_id} — using fast path")
                        else:
                            # Edge case: bound but failed previous embedding? Treat as registration attempt.
                            crop = self._crop_from_frame(frame, track['bbox'])
                            if crop is not None:
                                self.reid_queued.add(tid)
                                self.reid_queue.put((tid, crop, frame_number, True)) # is_reg=True

        # Drain phase — NO t3.join() here (deadlock). Just wait with timeout.
        logging.info("T2 draining remaining Re-ID results...")
        deadline = time.time() + 60
        while time.time() < deadline:
            while not self.reid_result_queue.empty():
                try:
                    tid, fid, emb, sim, is_new, rfn, crop = self.reid_result_queue.get_nowait()
                    self.reid_queued.discard(tid)
                    if fid is not None:
                        last = self._last_frame if self._last_frame is not None else np.zeros((100,100,3), dtype=np.uint8)
                        self.state_machine.bind_face_id(tid, fid, emb, is_new, rfn, last,
                                                        self.db, self.logger, self.session_id, self.io_queue, crop)
                except queue.Empty:
                    break
            if len(self.reid_queued) == 0 and self.reid_queue.empty() and self.reid_result_queue.empty():
                break
            time.sleep(0.1)

        logging.info("T2 tracker worker stopped.")

    def _crop_from_frame(self, frame: np.ndarray, bbox: list):
        """Crop directly from frame using track bbox. No IoU matching."""
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0]) - 10)
        y1 = max(0, int(bbox[1]) - 10)
        x2 = min(w, int(bbox[2]) + 10)
        y2 = min(h, int(bbox[3]) + 10)
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _reid_worker(self):
        """T3 — exits only when T2 is dead AND queue empty. NOT based on stop_event."""
        while True:
            try:
                tid, face_crop, frame_number, is_reg = self.reid_queue.get(timeout=0.5)
            except queue.Empty:
                if not self.t2.is_alive() and self.reid_queue.empty():
                    break
                continue

            logging.info(f"T3: got track {tid} crop={face_crop.shape} is_reg={is_reg}")
            embedding = self.embedder.generate_embedding(face_crop, is_registration=is_reg)
            if embedding is None:
                logging.info(f"T3: embedding None for track {tid}")
                self.reid_result_queue.put((tid, None, None, 0.0, False, frame_number, face_crop))
                continue

            self.logger.log_embedding_generated(tid, frame_number, quality_passed=True)
            fid, sim, is_new = self.recognizer.resolve_identity(tid, embedding, self.db, self.config)
            self.logger.log_recognition(fid, tid, sim, frame_number)
            logging.info(f"T3: track {tid} -> face_id={fid} is_new={is_new} sim={sim:.3f}")

            if is_new:
                self.io_queue.put({'type': 'update_embedding', 'face_id': fid,
                                   'embedding': embedding, 'sample_count': 1})

            self.reid_result_queue.put((tid, fid, embedding, sim, is_new, frame_number, face_crop))

        logging.info("T3 Re-ID worker stopped.")

    def _io_worker(self):
        """T4 — exits only when T2 AND T3 are dead AND queue empty."""
        while True:
            try:
                task = self.io_queue.get(timeout=0.5)
            except queue.Empty:
                if not self.t2.is_alive() and not self.t3.is_alive() and self.io_queue.empty():
                    break
                continue

            try:
                t = task['type']
                if t == 'save_image':
                    os.makedirs(os.path.dirname(task['path']), exist_ok=True)
                    # Use frame-based timestamp for video consistency
                    event_type = task.get('event_type', 'event')
                    fn = task.get('frame_number', 0)
                    image_with_timestamp = self._add_timestamp(task['crop'], event_type, fn)
                    cv2.imwrite(task['path'], image_with_timestamp)
                elif t == 'log_event':
                    eid = self.db.log_event_pending(task['face_id'], task['event_type'],
                                                    task['image_path'], task['track_id'],
                                                    task['similarity'], task['frame_number'],
                                                    task['session_id'])
                    self.db.complete_event(eid)
                elif t == 'update_embedding':
                    self.db.update_embedding(task['face_id'], task['embedding'], task['sample_count'])
                elif t == 'increment_visitor':
                    cnt = self.db.increment_visitor_count(task['session_id'])
                    self.logger.log_face_registered(task['face_id'], task['track_id'],
                                                    cnt, task['frame_number'])
            except Exception as e:
                logging.error(f"T4 error on {task.get('type')}: {e}")

        logging.info("T4 I/O worker stopped.")

    def _add_timestamp(self, image: np.ndarray, event_type: str, frame_number: int) -> np.ndarray:
        """
        Adds a timestamp footer BELOW the image. 
        Uses video-relative time (frame_number / 25 FPS).
        """
        try:
            if image is None or image.size == 0:
                return image
            
            # 1. Ensure minimum width for timestamp
            h, w = image.shape[:2]
            target_w = max(160, w)
            if target_w > w:
                new_h = int(h * (target_w / w))
                img = cv2.resize(image, (target_w, new_h), interpolation=cv2.INTER_AREA)
                h, w = img.shape[:2]
            else:
                img = image.copy()

            # 2. Add padding at the bottom (solid black footer)
            footer_h = 35
            canvas = np.zeros((h + footer_h, w, 3), dtype=np.uint8)
            canvas[0:h, 0:w] = img
            
            # 3. Calculate video relative time (MM:SS)
            # Assuming 25 FPS based on tracker config
            total_seconds = int(frame_number / 25)
            mm = total_seconds // 60
            ss = total_seconds % 60
            label = f"{event_type.upper()} {mm:02d}:{ss:02d}"
            
            cv2.putText(
                canvas, label, (8, h + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
            )
            return canvas
        except Exception as e:
            logging.warning(f"Timestamp error for {event_type} at frame {frame_number}: {e}")
            return image