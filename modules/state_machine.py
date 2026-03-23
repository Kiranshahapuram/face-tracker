"""
State machine module.
Manages 5-state tracking transitions per face.

States: UNSEEN → PENDING_ENTRY → IN_FRAME → GRACE_PERIOD → EXITED

KEY FIX: Entry and exit events are now triggered by zone line crossing,
not just by track birth/death. This prevents edge-of-frame false events
and correctly counts people walking through the monitored area.

Zone lines are horizontal (y-coordinate based):
  - entry_line_y: crossing downward (top of frame → into scene) = ENTRY
  - exit_line_y:  crossing downward (into scene → bottom) = EXIT
  - Direction is determined by centroid crossing the line between frames.
"""

from dataclasses import dataclass, field
import numpy as np
import time
import os
import cv2
import logging
from datetime import datetime


@dataclass
class FaceState:
    face_id: str
    state: str
    track_id: int
    pending_frames: int
    grace_frames: int
    last_bbox: list
    last_centroid_y: float       # track centroid y in previous frame — for zone crossing
    entry_fired: bool            # guard: only fire entry once per visit
    last_embedding: np.ndarray | None = None
    last_face_crop: np.ndarray | None = None  # add this


class FaceStateMachine:
    def __init__(self, config):
        self.config = config
        self.entry_line_y = config.tracking.entry_line_y
        self.exit_line_y  = config.tracking.exit_line_y
        # face_id → FaceState for bound tracks
        self.states: dict[str, FaceState] = {}
        # track_id → face_id for bound tracks
        self.track_to_face: dict[int, str] = {}
        # track_id → FaceState for unbound tracks (Re-ID not yet returned)
        self.temp_states: dict[int, FaceState] = {}
        # session-level set to prevent double entry for same identity
        self._entered_face_ids: set[str] = set()

    def _centroid_y(self, bbox: list) -> float:
        return (bbox[1] + bbox[3]) / 2.0

    def _crossed_entry_line(self, prev_y: float, curr_y: float) -> bool:
        """Returns True if centroid crossed entry_line_y going downward (top→bottom)."""
        return prev_y < self.entry_line_y <= curr_y

    def _crossed_exit_line(self, prev_y: float, curr_y: float) -> bool:
        """Returns True if centroid crossed exit_line_y going downward."""
        return prev_y < self.exit_line_y <= curr_y

    def _below_entry_line(self, bbox: list) -> bool:
        """True if centroid is already below the entry line (person entered from off-screen top)."""
        return self._centroid_y(bbox) >= self.entry_line_y

    def update(self, active_tracks: list[dict], frame_number: int, frame: np.ndarray,
               recognizer, embedder, db, logger, session_id: str, io_queue) -> list[dict]:
        events_emitted = []
        current_track_ids = {t['track_id'] for t in active_tracks}

        # ── 1. Update active tracks ────────────────────────────────────────
        for track in active_tracks:
            track_id = track['track_id']
            bbox     = track['bbox']
            curr_y   = self._centroid_y(bbox)

            face_id    = self.track_to_face.get(track_id)
            state_obj  = None

            if face_id and face_id in self.states:
                state_obj = self.states[face_id]
            elif track_id in self.temp_states:
                state_obj = self.temp_states[track_id]
            else:
                # New track — UNSEEN → PENDING_ENTRY
                state_obj = FaceState(
                    face_id        = None,
                    state          = "PENDING_ENTRY",
                    track_id       = track_id,
                    pending_frames = 1,
                    grace_frames   = 0,
                    last_bbox      = bbox,
                    last_centroid_y= curr_y,
                    entry_fired    = False,
                    last_embedding = None,
                )
                self.temp_states[track_id] = state_obj
                logging.debug(f"New track {track_id} → PENDING_ENTRY at y={curr_y:.0f}")

            prev_y = state_obj.last_centroid_y
            state_obj.last_bbox       = bbox
            state_obj.last_centroid_y = curr_y

            if state_obj.state == "PENDING_ENTRY":
                state_obj.pending_frames += 1
                if state_obj.pending_frames >= self.config.tracking.min_entry_frames:
                    state_obj.state = "IN_FRAME"
                    logging.debug(f"Track {track_id} → IN_FRAME after {state_obj.pending_frames} frames")

                # Zone line crossing check — ENTRY
                # FIX: entry fires on zone line crossing, not on track birth
                if not state_obj.entry_fired and state_obj.face_id is not None:
                    if state_obj.face_id not in self._entered_face_ids:
                        if self._crossed_entry_line(prev_y, curr_y) or self._below_entry_line(bbox):
                            state_obj.entry_fired = True
                            self._entered_face_ids.add(state_obj.face_id)
                            events_emitted.extend(
                                self._fire_entry(state_obj, track_id, frame_number, frame, db, logger, session_id, io_queue)
                            )
                    else:
                        # Already entered this session — just mark fired to prevent future checks
                        state_obj.entry_fired = True
                        logging.debug(f"face_id={state_obj.face_id} already entered this session, skipping duplicate entry")

                # Zone line crossing check — EXIT
                if state_obj.face_id is not None and self._crossed_exit_line(prev_y, curr_y):
                    state_obj.state = "EXITED"
                    events_emitted.extend(
                        self._fire_exit(state_obj, track_id, frame_number, frame, db, logger, recognizer, session_id, io_queue)
                    )
                    self._cleanup(state_obj)
                    continue

                # Periodic tracking log
                if frame_number % 10 == 0 and state_obj.face_id is not None:
                    logger.log_tracking(state_obj.face_id, track_id, frame_number)

            elif state_obj.state == "GRACE_PERIOD":
                # Track reappeared — return to IN_FRAME
                state_obj.state       = "IN_FRAME"
                state_obj.grace_frames = 0
                logging.debug(f"Track {track_id} recovered from GRACE_PERIOD")

        # ── 2. Handle disappeared tracks ──────────────────────────────────
        all_states = list(self.states.values()) + list(self.temp_states.values())
        for state_obj in all_states:
            if state_obj.track_id in current_track_ids:
                continue
            if state_obj.state == "EXITED":
                continue

            if state_obj.state in ("IN_FRAME", "PENDING_ENTRY"):
                state_obj.state        = "GRACE_PERIOD"
                state_obj.grace_frames = 1

            elif state_obj.state == "GRACE_PERIOD":
                state_obj.grace_frames += 1
                if state_obj.grace_frames > self.config.tracking.max_track_age_frames:
                    # Grace expired — fire exit if we had a bound identity and entry was fired
                    if state_obj.face_id is not None and state_obj.entry_fired:
                        state_obj.state = "EXITED"
                        events_emitted.extend(
                            self._fire_exit(state_obj, state_obj.track_id, frame_number, frame,
                                            db, logger, recognizer, session_id, io_queue)
                        )
                    self._cleanup(state_obj)

        return events_emitted

    def bind_face_id(self, track_id, face_id, embedding, is_new, frame_number, frame, db, logger, session_id, io_queue, face_crop=None):
        events = []

        if track_id in self.temp_states:
            state_obj = self.temp_states.pop(track_id)
        elif track_id in self.track_to_face:
            # Already bound — no new entry needed
            return events
        else:
            # Track expired during Re-ID — person was real, still register them
            logging.info(f"Track {track_id} expired before bind — creating entry for face {face_id}")
            state_obj = FaceState(
                face_id=None, state="IN_FRAME", track_id=track_id,
                pending_frames=3, grace_frames=0,
                last_bbox=[0, 0, 50, 50],
                last_centroid_y=0.0,
                entry_fired=False,
                last_embedding=embedding,
                last_face_crop=face_crop,
            )

        state_obj.face_id = face_id
        state_obj.last_embedding = embedding
        state_obj.last_face_crop = face_crop

        if face_id in self.states:
            old = self.states[face_id]
            old.track_id = track_id
            old.state = state_obj.state
            old.last_bbox = state_obj.last_bbox
            old.last_centroid_y = state_obj.last_centroid_y
            old.pending_frames = state_obj.pending_frames
            state_obj = old
        else:
            self.states[face_id] = state_obj

        self.track_to_face[track_id] = face_id

        if not state_obj.entry_fired:
            if face_id not in self._entered_face_ids:
                state_obj.entry_fired = True
                self._entered_face_ids.add(face_id)
                events.extend(self._fire_entry(
                    state_obj, track_id, frame_number, frame,
                    db, logger, session_id, io_queue, is_new
                ))
            else:
                state_obj.entry_fired = True
                logging.debug(f"face_id={face_id} already entered (via bind), skipping duplicate")

        if is_new:
            io_queue.put({
                'type': 'increment_visitor',
                'session_id': session_id,
                'face_id': face_id,
                'track_id': track_id,
                'frame_number': frame_number,
            })

        return events

    # ─────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────
    def _crop_from_bbox(self, frame: np.ndarray, bbox: list, padding: int = 10) -> np.ndarray:
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0]) - padding)
        y1 = max(0, int(bbox[1]) - padding)
        x2 = min(w, int(bbox[2]) + padding)
        y2 = min(h, int(bbox[3]) + padding)
        return frame[y1:y2, x1:x2]

    def _image_path(self, event_type: str, face_id: str, track_id: int, log_dir: str) -> str:
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        ms = str(int(time.time() * 1000))[-3:]
        folder = os.path.join(log_dir, f"{event_type}s", today)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f"{time_str}_{ms}_track{track_id}_{face_id}.jpg")

    def _fire_entry(self, state_obj, track_id, frame_number, frame,
                    db, logger, session_id, io_queue, is_new=False) -> list:
        img_path = self._image_path("entr", state_obj.face_id, track_id, self.config.system.log_dir)
        crop     = self._crop_from_bbox(frame, state_obj.last_bbox)

        # FIX: offload image save and DB write to T4 — never block T2
        if crop.size > 0:
            io_queue.put({'type': 'save_image', 'path': img_path, 'crop': crop, 'event_type': 'entry', 'frame_number': frame_number})
        io_queue.put({
            'type':         'log_event',
            'face_id':      state_obj.face_id,
            'event_type':   'entry',
            'image_path':   img_path,
            'track_id':     track_id,
            'similarity':   1.0,
            'frame_number': frame_number,
            'session_id':   session_id,
        })

        logger.log_entry(state_obj.face_id, track_id, frame_number, img_path)
        return [{'type': 'entry', 'face_id': state_obj.face_id, 'track_id': track_id, 'frame': frame_number}]

    def _fire_exit(self, state_obj, track_id, frame_number, frame,
                   db, logger, recognizer, session_id, io_queue) -> list:
        img_path = self._image_path("exit", state_obj.face_id, track_id, self.config.system.log_dir)
        # Use last known good face crop — current frame at exit time shows empty space
        if hasattr(state_obj, 'last_face_crop') and state_obj.last_face_crop is not None:
            # last_face_crop is already cropped from a previous frame
            exit_crop = state_obj.last_face_crop
        else:
            exit_crop = self._crop_from_bbox(frame, state_obj.last_bbox)

        if exit_crop.size > 0:
            io_queue.put({'type': 'save_image', 'path': img_path, 'crop': exit_crop, 'event_type': 'exit', 'frame_number': frame_number})
        io_queue.put({
            'type':         'log_event',
            'face_id':      state_obj.face_id,
            'event_type':   'exit',
            'image_path':   img_path,
            'track_id':     track_id,
            'similarity':   1.0,
            'frame_number': frame_number,
            'session_id':   session_id,
        })

        logger.log_exit(state_obj.face_id, track_id, frame_number, img_path)
        recognizer.release_track(track_id)
        if state_obj.last_embedding is not None:
            recognizer.add_to_reentry_buffer(state_obj.face_id, state_obj.last_embedding)

        return [{'type': 'exit', 'face_id': state_obj.face_id, 'track_id': track_id, 'frame': frame_number}]

    def _cleanup(self, state_obj):
        """Remove state from all dicts after EXITED."""
        if state_obj.face_id and state_obj.face_id in self.states:
            del self.states[state_obj.face_id]
        if state_obj.track_id in self.temp_states:
            del self.temp_states[state_obj.track_id]
        if state_obj.track_id in self.track_to_face:
            del self.track_to_face[state_obj.track_id]
