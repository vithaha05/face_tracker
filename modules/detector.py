import cv2
import json
import logging
import os
import numpy as np
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceDetector")

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

class FaceDetector:
    """
    A class for real-time face detection using a YOLOv8 model.
    """
    def __init__(self, config_path: str = "config.json"):
        """
        Initializes the FaceDetector by loading configuration and the YOLO model.
        
        :param config_path: Path to the config.json file.
        """
        config = load_config(config_path)
        self.model_path = config.get("yolo_model_path", "yolov8n-face.pt")
        self.conf_threshold = config.get("detection_confidence", 0.5)
        self.frame_skip = config.get("frame_skip", 3)
        
        try:
            if not os.path.exists(self.model_path):
                logger.info(f"Downloading/Loading YOLO model from {self.model_path}...")
            
            # Load the model
            self.model = YOLO(self.model_path)
            logger.info(f"YOLO model loaded successfully from {self.model_path}.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def should_detect(self, frame_number: int) -> bool:
        """
        Determines if detection should be performed on the current frame based on frame_skip.
        
        :param frame_number: The current frame index.
        :return: True if detection should be performed, False otherwise.
        """
        should = (frame_number % self.frame_skip == 0)
        if not should:
            logger.debug(f"Skipping detection for frame {frame_number}.")
        return should

    def detect_faces(self, frame: np.ndarray) -> list[dict]:
        """
        Runs YOLO inference on a BGR frame to detect faces.
        
        :param frame: The BGR frame as a NumPy array.
        :return: A list of dictionaries, each containing 'bbox' and 'confidence'.
        """
        if frame is None:
            return []

        h, w = frame.shape[:2]
        detections = []

        try:
            # Run inference
            results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get bbox coordinates (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    
                    # Clip to frame boundaries and convert to integers
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(w, int(x2))
                    y2 = min(h, int(y2))
                    
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf
                    })
            
            return detections
        except Exception as e:
            logger.error(f"Error during face detection inference: {e}")
            return []

    def crop_face(self, frame: np.ndarray, bbox: list[int], padding: int = 10) -> np.ndarray:
        """
        Crops a face from the frame based on a bbox with optional padding.
        
        :param frame: The BGR frame as a NumPy array.
        :param bbox: A list of [x1, y1, x2, y2] coordinates.
        :param padding: Padding pixels to add around the bbox.
        :return: The cropped face as a NumPy array.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Apply padding and clip to frame boundaries
        px1 = max(0, x1 - padding)
        py1 = max(0, y1 - padding)
        px2 = min(w, x2 + padding)
        py2 = min(h, y2 + padding)
        
        return frame[py1:py2, px1:px2]

if __name__ == "__main__":
    # Standard testing block
    try:
        detector = FaceDetector()
        
        # Create a dummy blank frame for testing if no camera/image is available
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a white rectangle to simulate a potential region (YOLO won't detect it as a face, though)
        cv2.rectangle(dummy_frame, (200, 200), (300, 300), (255, 255, 255), -1)
        
        logger.info("Running standalone test on dummy frame...")
        faces = detector.detect_faces(dummy_frame)
        
        print(f"Number of faces detected: {len(faces)}")
        for i, face in enumerate(faces):
            print(f"Face {i+1}: bbox={face['bbox']}, confidence={face['confidence']:.2f}")
            
    except Exception as e:
        logger.error(f"Test block failed: {e}")
