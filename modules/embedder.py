"""
Face embedding module.
Bypasses InsightFace internal face detector entirely.
YOLO is the detector. InsightFace is the embedder only.
"""

import cv2
import numpy as np
import logging
from insightface.app import FaceAnalysis


class FaceEmbedder:
    def __init__(self, config):
        self.config = config
        logging.info("Loading InsightFace model...")
        self.app = FaceAnalysis(
            name=self.config.reid.insightface_model,
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=-1, det_size=(160, 160))

        # Get recognition model directly — bypasses internal face detector
        self.rec_model = None
        for model in self.app.models.values():
            if hasattr(model, 'get_feat'):
                self.rec_model = model
                logging.info(f"Found recognition model: {type(model).__name__}")
                break

        if self.rec_model is None:
            logging.warning("Could not find rec model with get_feat — will use app.get() fallback")

        logging.info("InsightFace model loaded.")

    def quality_gate(self, face_crop: np.ndarray) -> tuple[bool, str]:
        min_size = self.config.detection.min_face_size_px
        if face_crop.shape[0] < min_size or face_crop.shape[1] < min_size:
            return False, f"too_small({face_crop.shape[1]}x{face_crop.shape[0]})"
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < self.config.quality_gate.min_blur_score:
            return False, f"blurry({blur_score:.1f})"
        return True, "passed"

    def generate_embedding(self, face_crop: np.ndarray) -> np.ndarray | None:
        try:
            if face_crop is None or face_crop.size == 0:
                logging.info("EMBED: crop is None/empty")
                return None

            logging.info(f"EMBED: crop shape={face_crop.shape}")

            passed, reason = self.quality_gate(face_crop)
            logging.info(f"EMBED: quality={passed} reason={reason}")
            if not passed:
                return None

            # PRIMARY: use recognition model directly — no internal face detection
            if self.rec_model is not None:
                try:
                    resized = cv2.resize(face_crop, (112, 112))
                    emb = self.rec_model.get_feat([resized])
                    logging.info(f"EMBED: get_feat returned type={type(emb)}")
                    if emb is not None and len(emb) > 0:
                        result = np.array(emb[0], dtype=np.float32)
                        norm = np.linalg.norm(result)
                        if norm > 0:
                            result = result / norm
                        logging.info(f"EMBED: SUCCESS get_feat shape={result.shape} norm={norm:.3f}")
                        return result
                    else:
                        logging.info("EMBED: get_feat returned empty")
                except Exception as e:
                    logging.info(f"EMBED: get_feat exception: {e}")

            # FALLBACK: try app.get() at multiple sizes
            for size in [(112, 112), (160, 160), (224, 224)]:
                try:
                    resized = cv2.resize(face_crop, size)
                    faces = self.app.get(resized)
                    logging.info(f"EMBED: app.get() at {size} found {len(faces) if faces else 0} faces")
                    if faces:
                        emb = faces[0].normed_embedding
                        logging.info(f"EMBED: SUCCESS app.get() at {size}")
                        return np.array(emb, dtype=np.float32)
                except Exception as e:
                    logging.info(f"EMBED: app.get() at {size} exception: {e}")

            logging.info(f"EMBED: ALL paths failed crop={face_crop.shape}")
            return None

        except Exception as e:
            logging.error(f"Embedder critical error: {e}")
            return None

    def update_running_mean(self, existing: np.ndarray, new: np.ndarray, n: int) -> np.ndarray:
        try:
            updated = ((existing * n) + new) / (n + 1)
            norm = np.linalg.norm(updated)
            if norm == 0:
                return updated
            return (updated / norm).astype(np.float32)
        except Exception as e:
            logging.error(f"Running mean error: {e}")
            return existing