import os
import json
import logging
import sqlite3
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from insightface.app import FaceAnalysis

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceRecognizer")

def load_config(config_path: str = "config.json") -> dict:
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read config file: {e}")
    return {}

class FaceRecognizer:
    """
    A class for face recognition, managing embeddings, and profile updates.
    Implements Fixes 2, 3, and 4 (Multi-embedding, Online updates, Confirmation).
    """
    def __init__(self, config_path: str = "config.json"):
        config = load_config(config_path)
        self.similarity_threshold = config.get("similarity_threshold", 0.5)
        self.model_name = config.get("model_name", "buffalo_l")
        self.db_path = config.get("db_path", "faces_db/faces.db")
        self.faces_dir = "faces_db/face_images"
        
        # New Re-ID params
        self.max_embeddings = config.get("max_embeddings_per_face", 5)
        self.conf_frames = config.get("embedding_confirmation_frames", 5)
        
        os.makedirs(self.faces_dir, exist_ok=True)
        
        # Initialize InsightFace
        try:
            self.app = FaceAnalysis(name=self.model_name, providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info(f"InsightFace model '{self.model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load InsightFace: {e}")
            raise

        # Database connection
        from modules.database import Database
        self.db = Database(self.db_path)
        
        # In-memory storage: {face_id: [embedding, embedding, ...]} (Fix 2)
        self.known_embeddings: Dict[str, List[np.ndarray]] = {}
        self.load_stored_embeddings()
        
        # Buffer for unconfirmed faces: {key: count} (Fix 4)
        self.unconfirmed_faces: Dict[str, int] = {}
        
        logger.info(f"Re-ID Fixes initialized: threshold={self.similarity_threshold}, "
                    f"multi-embedding=True, confirmation_frames={self.conf_frames}")

    def load_stored_embeddings(self):
        """
        Loads all known embeddings from the DB and groups them by face_id.
        """
        try:
            all_embs = self.db.get_all_embeddings() # Dict[face_id, List[np.ndarray]]
            self.known_embeddings = all_embs
            logger.info(f"Loaded {len(self.known_embeddings)} known faces from DB.")
        except Exception as e:
            logger.error(f"Failed to load embeddings from DB: {e}")

    def get_embedding(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts a 512-dim embedding from a face crop.
        """
        if face_crop is None or face_crop.size == 0:
            return None
        
        try:
            # If the crop is very small, we resize it to help InsightFace's internal detector
            h, w = face_crop.shape[:2]
            if h < 160 or w < 160:
                face_crop = cv2.resize(face_crop, (256, 256), interpolation=cv2.INTER_CUBIC)

            faces = self.app.get(face_crop)
            if not faces:
                # Still failed? It might be too blurry or not a face
                return None
            
            # Return the embedding of the largest face found in the crop
            faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
            return faces[0].embedding
        except Exception as e:
            logger.error(f"Error during embedding extraction: {e}")
            return None

    def match_face(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        FIX 2: Compares current embedding against ALL stored embeddings for each person.
        Returns the face_id with the highest similarity score.
        """
        best_match_id = None
        highest_score = -1.0
        
        for face_id, stored_list in self.known_embeddings.items():
            # Check all embeddings for this person and take the best
            # (stored_list contains normalized vectors, so DOT product == Cosine Similarity)
            scores = [np.dot(embedding, stored_emb) for stored_emb in stored_list]
            best_person_score = max(scores) if scores else 0.0
            
            if best_person_score > highest_score:
                highest_score = best_person_score
                best_match_id = face_id
                
        return best_match_id, highest_score

    def update_embedding(self, face_id: str, new_embedding: np.ndarray):
        """
        FIX 3: Online updates. Refreshes the profile over time to handle pose/lighting changes.
        """
        if face_id not in self.known_embeddings:
            return
            
        # Simple moving average to smooth transitions
        old_embedding = self.known_embeddings[face_id][-1]
        refreshed = (old_embedding + new_embedding) / 2.0
        refreshed = refreshed / np.linalg.norm(refreshed)
        
        # Add to the list and rotate if necessary (Fix 2)
        self.known_embeddings[face_id].append(refreshed)
        if len(self.known_embeddings[face_id]) > self.max_embeddings:
            self.known_embeddings[face_id].pop(0) # Keep most recent
            
        # Persist to database
        self.db.insert_embedding(face_id, refreshed)

    def identify_or_register(self, face_crop: np.ndarray, tracker_id: int = None) -> Optional[str]:
        """
        FIXES 2, 3, 4: Integrated identification with confirmation and multi-embedding storage.
        """
        embedding = self.get_embedding(face_crop)
        if embedding is None:
            return None
            
        face_id, score = self.match_face(embedding)
        
        if face_id and score >= self.similarity_threshold:
            # Match: Update profile (Fix 3)
            self.update_embedding(face_id, embedding)
            # Remove from confirmation if it was transiently new
            if tracker_id is not None:
                self.unconfirmed_faces.pop(str(tracker_id), None)
            return face_id
            
        # No Match: Buffer until confirmed (Fix 4)
        # Use tracker_id as a very stable key for confirmation
        key = str(tracker_id) if tracker_id is not None else self._embedding_key(embedding)
        self.unconfirmed_faces[key] = self.unconfirmed_faces.get(key, 0) + 1
        
        if self.unconfirmed_faces[key] >= self.conf_frames:
            # Confirmed: Register new profile (Fix 1-2)
            new_id = f"person_{len(self.known_embeddings) + 1}"
            self.register_face_with_id(new_id, embedding, face_crop)
            self.unconfirmed_faces.pop(key, None)
            logger.info(f"Registered new Face: {new_id} after {self.conf_frames} frames context (key={key}).")
            return new_id
            
        return None

    def _embedding_key(self, embedding: np.ndarray) -> str:
        """Fallback key if tracker_id is not available."""
        return str(embedding[:8].round(1).tolist())

    def register_face_with_id(self, face_id: str, embedding: np.ndarray, face_crop: np.ndarray):
        """Registers a face ID and its initial embedding."""
        if self.db.insert_face(face_id, embedding):
            self.db.insert_embedding(face_id, embedding)
            self.known_embeddings[face_id] = [embedding]
            
            # Save visual reference
            if face_crop is not None:
                path = os.path.join(self.faces_dir, f"{face_id}.jpg")
                cv2.imwrite(path, face_crop)
            return True
        return False
