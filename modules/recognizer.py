import os
import json
import logging
import sqlite3
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from insightface.app import FaceAnalysis

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceRecognizer")

from modules.utils import load_config

class FaceRecognizer:
    def __init__(self, config_path: str = "config.json"):
        config = load_config(config_path)
        self.similarity_threshold = config.get("similarity_threshold", 0.35)
        self.model_name = config.get("model_name", "buffalo_l")
        self.db_path = config.get("db_path", "faces_db/faces.db")
        self.max_embeddings = config.get("max_embeddings_per_face", 10)
        self.conf_frames = config.get("embedding_confirmation_frames", 2)
        self.debug_mode = config.get("debug_mode", True)
        
        self.faces_dir = "faces_db/face_images"
        os.makedirs(self.faces_dir, exist_ok=True)
        
        try:
            self.app = FaceAnalysis(name=self.model_name, providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info(f"InsightFace model '{self.model_name}' loaded.")
        except Exception as e:
            logger.error(f"Failed to load InsightFace: {e}")
            raise

        from modules.database import Database
        self.db = Database(self.db_path)
        self.known_embeddings: Dict[str, List[np.ndarray]] = {}
        self.load_stored_embeddings()
        self.unconfirmed_faces: Dict[str, int] = {}
        
    def load_stored_embeddings(self):
        try:
            self.known_embeddings = self.db.get_all_embeddings()
            logger.info(f"Loaded {len(self.known_embeddings)} faces from DB.")
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")

    def get_embedding(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """FIX 3: 4-stage fallback (direct -> padded -> 112 -> 160)."""
        if face_crop is None or face_crop.size == 0:
            return None
        
        # Ensure min size for robust detection
        h, w = face_crop.shape[:2]
        if h < 112 or w < 112:
            face_crop = cv2.resize(face_crop, (112, 112))

        # Stage 1: Direct
        faces = self.app.get(face_crop)
        if faces:
            emb = faces[0].embedding
            return emb / np.linalg.norm(emb)

        # Stage 2: Padded (InsightFace often likes context)
        padded = cv2.copyMakeBorder(face_crop, 40, 40, 40, 40, cv2.BORDER_REPLICATE)
        faces = self.app.get(padded)
        if faces:
            emb = faces[0].embedding
            return emb / np.linalg.norm(emb)

        # Stage 3: Resized 112x112 (InsightFace standard)
        res112 = cv2.resize(face_crop, (112, 112))
        faces = self.app.get(res112)
        if faces:
            emb = faces[0].embedding
            return emb / np.linalg.norm(emb)
            
        # Stage 4: Resized 160x160 (Alternative standard)
        res160 = cv2.resize(face_crop, (160, 160))
        faces = self.app.get(res160)
        if faces:
            emb = faces[0].embedding
            return emb / np.linalg.norm(emb)
            
        return None

    def match_face(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """FIX 5: Detailed logging."""
        if not self.known_embeddings:
            return None, 0.0

        best_id = None
        best_score = 0.0

        for face_id, embeddings_list in self.known_embeddings.items():
            for stored_emb in embeddings_list:
                score = float(np.dot(embedding, stored_emb))
                if score > best_score:
                    best_score = score
                    best_id = face_id

        if self.debug_mode:
            logger.debug(f"Match check: best_id={best_id} score={best_score:.4f} vs {self.similarity_threshold}")

        if best_score >= self.similarity_threshold:
            return best_id, best_score
        return None, best_score

    def update_embedding(self, face_id: str, new_embedding: np.ndarray):
        if face_id not in self.known_embeddings: return
        self.known_embeddings[face_id].append(new_embedding)
        if len(self.known_embeddings[face_id]) > self.max_embeddings:
            self.known_embeddings[face_id].pop(0)
        self.db.insert_embedding(face_id, new_embedding)

    def crop_helper(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """Standalone robust cropper."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        # Pad 40%
        pad_x = int((x2 - x1) * 0.4)
        pad_y = int((y2 - y1) * 0.4)
        nx1, ny1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        nx2, ny2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        crop = frame[ny1:ny2, nx1:nx2]
        if crop.shape[0] < 112 or crop.shape[1] < 112:
            crop = cv2.resize(crop, (112, 112))
        return crop

    def identify_or_register(self, detection: Dict[str, Any], frame: np.ndarray, event_logger=None, tracker_id: int = None, embedding: Optional[np.ndarray] = None) -> Optional[str]:
        """FIX 6: Clearer buffer logic with embedding reuse."""
        face_crop = None
        if embedding is None:
            bbox = detection.get("face_bbox") if detection.get("face_bbox") else detection.get("bbox")
            face_crop = self.crop_helper(frame, bbox)
            embedding = self.get_embedding(face_crop)
            if embedding is not None and event_logger:
                event_logger.log_embedding_generated(f"Track_{tracker_id}" if tracker_id else "unknown")
        
        if embedding is None:
            return None
            
        match_id, score = self.match_face(embedding)
        if match_id:
            if event_logger:
                event_logger.log_face_recognized(match_id, score)
            self.update_embedding(match_id, embedding)
            return match_id
            
        # Confirmation Buffer
        key = str(tracker_id) if tracker_id is not None else str(embedding[:8].sum())
        self.unconfirmed_faces[key] = self.unconfirmed_faces.get(key, 0) + 1
        
        if self.debug_mode: logger.debug(f"Confirmation {key}: {self.unconfirmed_faces[key]}/{self.conf_frames}")
        
        if self.unconfirmed_faces[key] >= self.conf_frames:
            self.unconfirmed_faces.pop(key, None)
            new_id = f"person_{len(self.known_embeddings) + 1}"
            if self.register_face_with_id(new_id, embedding, face_crop):
                logger.info(f"Registered {new_id} (Track {tracker_id})")
                return new_id
                
        return None

    def register_face_with_id(self, face_id: str, embedding: np.ndarray, face_crop: np.ndarray):
        if self.db.insert_face(face_id, embedding):
            self.known_embeddings[face_id] = [embedding]
            self.db.insert_embedding(face_id, embedding)
            if face_crop is not None:
                path = os.path.join(self.faces_dir, f"{face_id}.jpg")
                cv2.imwrite(path, face_crop)
            return True
        return False
