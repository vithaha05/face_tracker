import cv2
import json
import logging
import os
import uuid
import numpy as np
from typing import Optional, Union, Dict, Any, List
from insightface.app import FaceAnalysis
from modules.database import Database

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceRecognizer")

def load_config(config_path: str = "config.json") -> dict:
    """
    Reads the configuration file and returns a dictionary.
    
    :param config_path: Path to the config.json file.
    :return: Configuration dictionary or empty dict if not found.
    """
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read config file: {e}")
    return {}

class FaceRecognizer:
    """
    A class for face recognition and identity management using InsightFace and SQLite.
    """
    def __init__(self, config_path: str = "config.json"):
        """
        Initializes the FaceRecognizer, loads the model, and fetches existing embeddings.
        
        :param config_path: Path to the configuration file.
        """
        self.config = load_config(config_path)
        model_name = self.config.get("model_name", "buffalo_l")
        self.similarity_threshold = self.config.get("similarity_threshold", 0.6)
        
        # Database initialization
        self.db = Database(config_path)
        self.known_embeddings = {}
        
        # Faces directory for saving face crops
        self.faces_dir = os.path.dirname(self.db.db_path)
        os.makedirs(self.faces_dir, exist_ok=True)
        
        try:
            # Initialize InsightFace
            # providers: CUDA for GPU, CPU for fallback
            self.app = FaceAnalysis(name=model_name, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info(f"InsightFace model '{model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load InsightFace model '{model_name}': {e}")
            raise

        # Load existing identities
        self.load_stored_embeddings()

    def load_stored_embeddings(self) -> None:
        """
        Fetches all stored face embeddings from the database into the in-memory cache.
        """
        try:
            faces = self.db.get_all_embeddings()
            for face in faces:
                # The database module already returns np.ndarray from np.frombuffer
                self.known_embeddings[face["id"]] = face["embedding"]
            logger.info(f"Loaded {len(self.known_embeddings)} known faces from DB.")
        except Exception as e:
            logger.error(f"Error loading stored embeddings: {e}")

    def get_embedding(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Runs InsightFace on a cropped face image to extract a normalized embedding vector.
        
        :param face_crop: The cropped BGR face image.
        :return: A normalized unit-length NumPy array (512-dim) or None if detection fails.
        """
        if face_crop is None or face_crop.size == 0:
            return None

        try:
            # Run extraction
            faces = self.app.get(face_crop)
            if not faces:
                logger.warning("InsightFace failed to extract face from the crop.")
                return None
            
            # Use the most confident face in the crop (should be only one)
            face = faces[0]
            embedding = face.embedding.astype(np.float32)
            
            # Normalize to unit length
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding /= norm
            
            return embedding
        except Exception as e:
            logger.error(f"Error during embedding extraction: {e}")
            return None

    def match_face(self, embedding: np.ndarray) -> Optional[str]:
        """
        Compares an input embedding against the known store using cosine similarity.
        
        :param embedding: The normalized input embedding.
        :return: The face_id of the best match above the threshold, or None.
        """
        best_match_id = None
        best_similarity = -1.0
        
        for face_id, known_emb in self.known_embeddings.items():
            # embeddings are already unit-normalized, so dot product == cosine similarity
            similarity = np.dot(embedding, known_emb)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = face_id
                
        logger.debug(f"Best similarity: {best_similarity:.4f} (Threshold: {self.similarity_threshold})")
        
        if best_similarity >= self.similarity_threshold:
            logger.debug(f"Matched with identity {best_match_id}.")
            return best_match_id
        
        logger.debug("No match above threshold.")
        return None

    def register_face(self, embedding: np.ndarray, face_crop: np.ndarray) -> str:
        """
        Assigns a new UUID, saves the face image to disk, and persists the identity to DB.
        
        :param embedding: The normalized target embedding.
        :param face_crop: The cropped BGR image for storage.
        :return: The generated unique face_id.
        """
        face_id = str(uuid.uuid4())
        
        # Save face image to disk
        image_name = f"{face_id}.jpg"
        image_path = os.path.join(self.faces_dir, image_name)
        try:
            cv2.imwrite(image_path, face_crop)
        except Exception as e:
            logger.error(f"Failed to save face image for ID {face_id}: {e}")
            
        # Write to Database
        success = self.db.insert_face(face_id, embedding)
        if success:
            # Add to in-memory store
            self.known_embeddings[face_id] = embedding
            logger.info(f"New face registered: {face_id}")
        else:
            logger.error(f"Database insertion failed for face registration: {face_id}")
            
        return face_id

    def identify_or_register(self, face_crop: np.ndarray) -> Optional[str]:
        """
        The main entry point: extracts embedding, tries to match it, or registers if new.
        
        :param face_crop: The cropped BGR face image.
        :return: A face_id string or None if embedding failed.
        """
        embedding = self.get_embedding(face_crop)
        if embedding is None:
            return None
            
        face_id = self.match_face(embedding)
        if face_id:
            return face_id
            
        # No match found, register a new identity
        return self.register_face(embedding, face_crop)

if __name__ == "__main__":
    # Standard testing block
    try:
        recognizer = FaceRecognizer()
        
        # Create a dummy blank face-like crop for testing (insightface will likely fail on this)
        dummy_crop = np.zeros((160, 160, 3), dtype=np.uint8)
        cv2.rectangle(dummy_crop, (40, 40), (120, 120), (255, 255, 255), -1)
        
        logger.info("Running standalone test on dummy crop...")
        face_id = recognizer.identify_or_register(dummy_crop)
        
        print(f"Resulting Face ID: {face_id}")
        
    except Exception as e:
        logger.error(f"Standalone test block failed: {e}")
