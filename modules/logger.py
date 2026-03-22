import os
import json
import logging
import cv2
import numpy as np
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional, Union

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
            # Fallback to a very basic logging if config fails
            print(f"Failed to read config file in Logger: {e}")
    return {}

class EventLogger:
    """
    Manages structured logging for the face tracking system, including console logs,
    rotating event files, and image captures for entries/exits.
    """
    def __init__(self, config_path: str = "config.json"):
        """
        Initializes loggers and creates the necessary directory structure.
        
        :param config_path: Path to the config.json file.
        """
        config = load_config(config_path)
        self.log_dir = config.get("log_dir", "logs")
        
        # Create folder structure
        self.entries_dir = os.path.join(self.log_dir, "entries")
        self.exits_dir = os.path.join(self.log_dir, "exits")
        os.makedirs(self.entries_dir, exist_ok=True)
        os.makedirs(self.exits_dir, exist_ok=True)
        
        # 1. System Logger (Console)
        self.sys_logger = logging.getLogger("face_tracker")
        self.sys_logger.setLevel(logging.INFO)
        if not self.sys_logger.handlers:
            sys_handler = logging.StreamHandler()
            sys_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
            sys_handler.setFormatter(sys_formatter)
            self.sys_logger.addHandler(sys_handler)
            
        # 2. Event File Logger (File-only)
        self.event_logger = logging.getLogger("event_log")
        self.event_logger.setLevel(logging.INFO)
        self.event_logger.propagate = False # Prevent doubling to console
        
        event_file = os.path.join(self.log_dir, "events.log")
        if not self.event_logger.handlers:
            # 5MB max, 3 backups
            event_handler = RotatingFileHandler(event_file, maxBytes=5_000_000, backupCount=3)
            event_formatter = logging.Formatter('[%(asctime)s] %(message)s')
            event_handler.setFormatter(event_formatter)
            self.event_logger.addHandler(event_handler)
            
        self.log_system_event(f"EventLogger initialized. Log directory: {self.log_dir}")

    def _save_event_image(self, face_id: str, face_crop: Optional[np.ndarray], base_dir: str) -> Optional[str]:
        """
        Helper to save an entry/exit image to the date-organized directory structure.
        
        :param face_id: The ID of the face.
        :param face_crop: The BGR face crop array.
        :param base_dir: Parent directory (entries or exits).
        :return: Relative path to saved image or None.
        """
        if face_crop is None or face_crop.size == 0:
            self.sys_logger.warning(f"Face crop empty/None for face_id={face_id}. Skipping image save.")
            return None
            
        try:
            now = datetime.utcnow()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%Y%m%d_%H%M%S_%f")
            
            target_dir = os.path.join(base_dir, date_str)
            os.makedirs(target_dir, exist_ok=True)
            
            filename = f"{face_id}_{time_str}.jpg"
            save_path = os.path.join(target_dir, filename)
            
            success = cv2.imwrite(save_path, face_crop)
            if not success:
                self.sys_logger.warning(f"Failed to save image for face_id={face_id} to {save_path}")
                return None
            
            return save_path
        except Exception as e:
            self.sys_logger.error(f"Error saving event image for face_id={face_id}: {e}")
            return None

    def log_entry(self, face_id: str, face_crop: Optional[np.ndarray]) -> Optional[str]:
        """
        Logs a face entry event, saves the image, and updates the event log.
        
        :param face_id: The identified face ID.
        :param face_crop: The BGR face crop array.
        :return: Path to saved image or None.
        """
        image_path = self._save_event_image(face_id, face_crop, self.entries_dir)
        timestamp = datetime.utcnow().isoformat()
        
        event_msg = f"[ENTRY] face_id={face_id} image={image_path}"
        self.event_logger.info(event_msg)
        self.sys_logger.info(f"Entry logged: face_id={face_id}")
        
        return image_path

    def log_exit(self, face_id: str, face_crop: Optional[np.ndarray]) -> Optional[str]:
        """
        Logs a face exit event, saves the image, and updates the event log.
        
        :param face_id: The identified face ID.
        :param face_crop: The BGR face crop array.
        :return: Path to saved image or None.
        """
        image_path = self._save_event_image(face_id, face_crop, self.exits_dir)
        timestamp = datetime.utcnow().isoformat()
        
        event_msg = f"[EXIT] face_id={face_id} image={image_path}"
        self.event_logger.info(event_msg)
        self.sys_logger.info(f"Exit logged: face_id={face_id}")
        
        return image_path

    def log_embedding_generated(self, face_id: str) -> None:
        """
        Logs that a face embedding was generated.
        
        :param face_id: The associated face ID.
        """
        self.sys_logger.debug(f"[EMBEDDING_GENERATED] face_id={face_id}")

    def log_face_registered(self, face_id: str) -> None:
        """
        Logs a new face registration to both system and event logs.
        
        :param face_id: The newly generated face ID.
        """
        self.sys_logger.info(f"New face registered: face_id={face_id}")
        self.event_logger.info(f"[REGISTRATION] face_id={face_id}")

    def log_face_recognized(self, face_id: str, similarity: float) -> None:
        """
        Logs a successful face recognition match.
        
        :param face_id: The matched face ID.
        :param similarity: The calculated similarity score.
        """
        self.sys_logger.debug(f"[RECOGNITION] face_id={face_id} similarity={similarity:.4f}")

    def log_system_event(self, message: str, level: str = "info") -> None:
        """
        General purpose method for logging system events at various levels.
        
        :param message: The log message.
        :param level: Severity level ('debug', 'info', 'warning', 'error').
        """
        lvl = level.lower()
        if lvl == "debug":
            self.sys_logger.debug(message)
        elif lvl == "warning":
            self.sys_logger.warning(message)
        elif lvl == "error":
            self.sys_logger.error(message)
        else:
            self.sys_logger.info(message)

if __name__ == "__main__":
    # Test block
    try:
        logger_inst = EventLogger()
        dummy_face = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Test entry
        entry_path = logger_inst.log_entry("test_uuid_123", dummy_face)
        # Test exit
        exit_path = logger_inst.log_exit("test_uuid_123", dummy_face)
        # Test registration
        logger_inst.log_face_registered("test_uuid_123")
        
        print("\nVerifying events.log contents:")
        event_file_path = os.path.join(logger_inst.log_dir, "events.log")
        if os.path.exists(event_file_path):
            with open(event_file_path, 'r') as f:
                lines = f.readlines()
                for line in lines[-5:]:
                    print(line.strip())
        
    except Exception as e:
        # Standard system-logger will handle this if it was initialized, but fallback is print here
        print(f"Test block failed: {e}")
