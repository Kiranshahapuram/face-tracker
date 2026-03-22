"""
Two-layer Re-ID module.
Handles active track bindings and re-entry buffers to resolve visitor identities.
"""

import time
import numpy as np
import logging

class FaceRecognizer:
    def __init__(self, config):
        self.config = config
        self.track_bindings: dict[int, str] = {}
        self.reentry_buffer: list[dict] = []

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def resolve_identity(self, track_id: int, embedding: np.ndarray, db, config) -> tuple[str, float, bool]:
        # Step 1: Active binding
        if track_id in self.track_bindings:
            return self.track_bindings[track_id], 1.0, False
            
        threshold = config.reid.similarity_threshold
        
        # Step 2: Re-entry buffer
        now = time.time()
        best_score = -1.0
        best_face_id = None
        
        for entry in self.reentry_buffer:
            if now - entry['exit_time'] <= config.reid.reentry_window_s:
                score = self.cosine_similarity(embedding, entry['embedding'])
                if score >= threshold and score > best_score:
                    best_score = score
                    best_face_id = entry['face_id']
                    
        if best_face_id is not None:
            self.track_bindings[track_id] = best_face_id
            return best_face_id, best_score, False

        # Step 3: Global DB
        db_face_id, db_score = db.find_similar_face(embedding, threshold)
        if db_face_id is not None:
            self.track_bindings[track_id] = db_face_id
            # Note: We rely on caller to know it's not new and update embedding directly through DB
            return db_face_id, db_score, False
            
        # Step 4: New Face
        new_face_id = db.register_face(embedding)
        self.track_bindings[track_id] = new_face_id
        return new_face_id, 1.0, True

    def release_track(self, track_id: int) -> str | None:
        return self.track_bindings.pop(track_id, None)

    def add_to_reentry_buffer(self, face_id: str, embedding: np.ndarray):
        now = time.time()
        self.reentry_buffer.append({
            'face_id': face_id,
            'embedding': embedding,
            'exit_time': now
        })
        self.evict_expired_reentry()
        
        buffer_max = self.config.reid.reentry_max_buffer
        if len(self.reentry_buffer) > buffer_max:
            # Buffer is usually appended in chronological order
            self.reentry_buffer = self.reentry_buffer[-buffer_max:]

    def evict_expired_reentry(self):
        now = time.time()
        window = self.config.reid.reentry_window_s
        self.reentry_buffer = [e for e in self.reentry_buffer if now - e['exit_time'] <= window]
