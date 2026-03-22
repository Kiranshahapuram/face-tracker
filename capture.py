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

    def _capture_worker(self):
        source = self.config.video.source
        is_rtsp = source.startswith("rtsp://")
        cap = cv2.VideoCapture(source)
        frame_number = 0
        retries = 0

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if is_rtsp:
                    retries += 1
                    if retries > self.config.video.rtsp_max_retries:
                        logging.error("Max RTSP retries reached.")
                        break
                    time.sleep(self.config.video.rtsp_reconnect_delay_s)
                    cap = cv2.VideoCapture(source)
                    continue
                else:
                    logging.info("End of video file.")
                    break
            retries = 0
            frame_number += 1
            if self.config.video.frame_skip_interval > 0:
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

            detections = self.detector.detect(frame)
            active_tracks = self.tracker.update(detections, frame)

            # Drain Re-ID results
            while not self.reid_result_queue.empty():
                try:
                    tid, fid, emb, sim, is_new, rfn = self.reid_result_queue.get_nowait()
                    self.reid_queued.discard(tid)
                    if fid is not None:
                        self.state_machine.bind_face_id(tid, fid, emb, is_new, rfn, frame,
                                                        self.db, self.logger, self.session_id, self.io_queue)
                except queue.Empty:
                    break

            self.state_machine.update(active_tracks, frame_number, frame,
                                      self.recognizer, self.embedder, self.db,
                                      self.logger, self.session_id, self.io_queue)

            for track in active_tracks:
                tid = track['track_id']
                if tid in self.state_machine.temp_states:
                    sobj = self.state_machine.temp_states[tid]
                    if sobj.pending_frames >= self.config.tracking.min_entry_frames and tid not in self.reid_queued:
                        crop = self._crop_from_frame(frame, track['bbox'])
                        if crop is not None:
                            self.reid_queued.add(tid)
                            self.reid_queue.put((tid, crop, frame_number))
                            logging.info(f"T2: queued track {tid} for Re-ID crop={crop.shape}")
                elif tid in self.state_machine.track_to_face:
                    if frame_number % reid_interval == 0 and tid not in self.reid_queued:
                        crop = self._crop_from_frame(frame, track['bbox'])
                        if crop is not None:
                            self.reid_queued.add(tid)
                            self.reid_queue.put((tid, crop, frame_number))

        # Drain phase — NO t3.join() here (deadlock). Just wait with timeout.
        logging.info("T2 draining remaining Re-ID results...")
        deadline = time.time() + 60
        while time.time() < deadline:
            while not self.reid_result_queue.empty():
                try:
                    tid, fid, emb, sim, is_new, rfn = self.reid_result_queue.get_nowait()
                    self.reid_queued.discard(tid)
                    if fid is not None:
                        last = self._last_frame if self._last_frame is not None else np.zeros((100,100,3), dtype=np.uint8)
                        self.state_machine.bind_face_id(tid, fid, emb, is_new, rfn, last,
                                                        self.db, self.logger, self.session_id, self.io_queue)
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
                tid, face_crop, frame_number = self.reid_queue.get(timeout=0.5)
            except queue.Empty:
                if not self.t2.is_alive() and self.reid_queue.empty():
                    break
                continue

            logging.info(f"T3: got track {tid} crop={face_crop.shape}")
            embedding = self.embedder.generate_embedding(face_crop)
            if embedding is None:
                logging.info(f"T3: embedding None for track {tid}")
                self.reid_result_queue.put((tid, None, None, 0.0, False, frame_number))
                continue

            self.logger.log_embedding_generated(tid, frame_number, quality_passed=True)
            fid, sim, is_new = self.recognizer.resolve_identity(tid, embedding, self.db, self.config)
            self.logger.log_recognition(fid, tid, sim, frame_number)
            logging.info(f"T3: track {tid} -> face_id={fid} is_new={is_new} sim={sim:.3f}")

            if is_new:
                self.io_queue.put({'type': 'update_embedding', 'face_id': fid,
                                   'embedding': embedding, 'sample_count': 1})

            self.reid_result_queue.put((tid, fid, embedding, sim, is_new, frame_number))

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
                    cv2.imwrite(task['path'], task['crop'])
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