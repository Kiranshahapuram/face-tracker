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

        # Fast OpenCV eye detector for registration frontality check
        # Replaces slow app.get() call — runs in microseconds vs milliseconds
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        # Fallback cascade — better at angled/partially occluded eyes
        self.eye_cascade2 = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
        )
        if self.eye_cascade.empty() or self.eye_cascade2.empty():
            logging.warning("One or more eye cascades not loaded")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        logging.info("OpenCV Haar cascades loaded for fast registration gate.")

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

    def _has_frontal_face(self, face_crop: np.ndarray) -> bool:
        """
        Fast frontality check using OpenCV Haar cascade.
        Replaces InsightFace app.get() in registration path.
        Runs in microseconds vs InsightFace's milliseconds on CPU.
        Returns True if two eyes detected with valid horizontal separation.
        """
        try:
            if self.eye_cascade.empty():
                return True  # cascade not available — skip check, don't block

            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)  # normalize lighting for dark crops

            eyes = self.eye_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=2,
                minSize=(6, 6)
            )

            # Fallback to second cascade if first finds fewer than 2 eyes
            # Better at angled faces and partially occluded eyes
            if len(eyes) < 2:
                eyes = self.eye_cascade2.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=2,
                    minSize=(6, 6)
                )

            if len(eyes) < 2:
                logging.info(f"REGISTRATION BLOCKED: only {len(eyes)} eye(s) detected")
                return False

            # Sort eyes left to right by x coordinate
            eyes = sorted(eyes, key=lambda e: e[0])
            left_cx  = eyes[0][0] + eyes[0][2] // 2
            right_cx = eyes[1][0] + eyes[1][2] // 2

            eye_dist  = abs(right_cx - left_cx)
            eye_ratio = eye_dist / face_crop.shape[1]

            if eye_ratio < 0.20:
                logging.info(f"REGISTRATION BLOCKED: side profile eye_ratio={eye_ratio:.2f}")
                return False

            return True

        except Exception as e:
            logging.warning(f"Frontality check error: {e}")
            return True  # on error, don't block — fail open

    def generate_embedding(self, face_crop: np.ndarray, is_registration: bool = False) -> np.ndarray | None:
        try:
            if face_crop is None or face_crop.size == 0:
                logging.info("EMBED: crop is None/empty")
                return None

            logging.info(f"EMBED: crop shape={face_crop.shape} is_registration={is_registration}")

            passed, reason = self.quality_gate(face_crop)
            logging.info(f"EMBED: quality={passed} reason={reason}")
            if not passed:
                return None

            if is_registration:
                try:
                    if face_crop is None or face_crop.size == 0:
                        return None

                    # Size check
                    min_size = self.config.reid.min_registration_face_size_px
                    if face_crop.shape[0] < min_size or face_crop.shape[1] < min_size:
                        logging.info(f"REGISTRATION BLOCKED: too small {face_crop.shape}")
                        return None

                    # Resize to 112x112 for embedding
                    resized = cv2.resize(face_crop, (112, 112))

                    # Fast frontality check — OpenCV Haar cascade
                    # Blocks backs of heads, side profiles, dark crops
                    if not self._has_frontal_face(resized):
                        return None

                    # Generate embedding via fast get_feat — no InsightFace detection
                    if self.rec_model is None:
                        return None

                    emb = self.rec_model.get_feat([resized])
                    if emb is None or len(emb) == 0:
                        logging.info("REGISTRATION BLOCKED: get_feat returned empty")
                        return None

                    result = np.array(emb[0], dtype=np.float32)
                    norm = np.linalg.norm(result)
                    if norm > 0:
                        result = result / norm

                    logging.info(f"REGISTRATION PASSED: norm={norm:.3f}")
                    return result.astype(np.float32)

                except Exception as e:
                    logging.error(f"Registration embedding error: {e}")
                    return None

            else:
                # Fast matching path — no landmark check
                try:
                    if face_crop is None or face_crop.size == 0:
                        return None

                    passed, reason = self.quality_gate(face_crop)
                    if not passed:
                        return None

                    resized = cv2.resize(face_crop, (112, 112))

                    if self.rec_model is not None:
                        try:
                            # get_feat accepts a list — pass as single-item batch
                            emb = self.rec_model.get_feat([resized])
                            if emb is not None and len(emb) > 0:
                                result = np.array(emb[0], dtype=np.float32)
                                norm = np.linalg.norm(result)
                                if norm > 0:
                                    result = result / norm
                                return result
                        except Exception as e:
                            logging.debug(f"get_feat error: {e}")

                    return None

                except Exception as e:
                    logging.error(f"Embedder matching error: {e}")
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