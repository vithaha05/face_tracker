import os
import json
import logging
import cv2
import numpy as np
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional, Union

from modules.utils import load_config

class EventLogger:
    """
    Manages structured logging for the tracking system.
    Updated to handle full body crops in addition to face crops.
    """
    def __init__(self, config_path: str = "config.json"):
        config = load_config(config_path)
        self.log_dir = config.get("log_dir", "logs")
        
        # Create folder structure
        self.entries_dir = os.path.join(self.log_dir, "entries")
        self.exits_dir = os.path.join(self.log_dir, "exits")
        os.makedirs(self.entries_dir, exist_ok=True)
        os.makedirs(self.exits_dir, exist_ok=True)
        
        # 1. System Logger
        self.sys_logger = logging.getLogger("face_tracker")
        self.sys_logger.setLevel(logging.INFO)
        if not self.sys_logger.handlers:
            sys_handler = logging.StreamHandler()
            sys_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
            sys_handler.setFormatter(sys_formatter)
            self.sys_logger.addHandler(sys_handler)
            
        # 2. Event File Logger
        self.event_logger = logging.getLogger("event_log")
        self.event_logger.setLevel(logging.INFO)
        self.event_logger.propagate = False 
        
        event_file = os.path.join(self.log_dir, "events.log")
        if not self.event_logger.handlers:
            event_handler = RotatingFileHandler(event_file, maxBytes=5_000_000, backupCount=3)
            event_formatter = logging.Formatter('[%(asctime)s] %(message)s')
            event_handler.setFormatter(event_formatter)
            self.event_logger.addHandler(event_handler)
            
        self.log_system_event(f"EventLogger initialized. Log directory: {self.log_dir}")

    def _save_event_image(self, face_id: str, crop: Optional[np.ndarray], base_dir: str) -> Optional[str]:
        """Helper to save an image crop."""
        if crop is None or crop.size == 0:
            return None
            
        try:
            now = datetime.utcnow()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%Y%m%d_%H%M%S_%f")
            
            target_dir = os.path.join(base_dir, date_str)
            os.makedirs(target_dir, exist_ok=True)
            
            filename = f"{face_id}_{time_str}.jpg"
            save_path = os.path.join(target_dir, filename)
            
            success = cv2.imwrite(save_path, crop)
            return save_path if success else None
        except Exception as e:
            self.sys_logger.error(f"Error saving image for {face_id}: {e}")
            return None

    def log_entry(self, face_id: str, crop: Optional[np.ndarray]) -> Optional[str]:
        """Logs a person entry event (saves body or face crop)."""
        image_path = self._save_event_image(face_id, crop, self.entries_dir)
        event_msg = f"[ENTRY] face_id={face_id} image={image_path}"
        self.event_logger.info(event_msg)
        self.sys_logger.info(f"Entry logged: face_id={face_id}")
        return image_path

    def log_exit(self, face_id: str, crop: Optional[np.ndarray]) -> Optional[str]:
        """Logs a person exit event."""
        image_path = self._save_event_image(face_id, crop, self.exits_dir)
        event_msg = f"[EXIT] face_id={face_id} image={image_path}"
        self.event_logger.info(event_msg)
        self.sys_logger.info(f"Exit logged: face_id={face_id}")
        return image_path

    def log_face_registered(self, face_id: str) -> None:
        """Logs a new face registration."""
        self.sys_logger.info(f"New face registered: face_id={face_id}")
        self.event_logger.info(f"[REGISTRATION] face_id={face_id}")

    def log_embedding_generated(self, face_id: str) -> None:
        """Logs that a face embedding was generated."""
        self.sys_logger.debug(f"Embedding generated for face_id={face_id}")
        self.event_logger.info(f"[EMBEDDING] face_id={face_id}")

    def log_face_recognized(self, face_id: str, score: float) -> None:
        """Logs a successful face recognition match."""
        self.sys_logger.debug(f"Face recognized: face_id={face_id} score={score:.4f}")
        self.event_logger.info(f"[RECOGNITION] face_id={face_id} score={score:.4f}")

    def log_system_event(self, message: str, level: str = "info") -> None:
        """General purpose log."""
        lvl = level.lower()
        if lvl == "debug": self.sys_logger.debug(message)
        elif lvl == "warning": self.sys_logger.warning(message)
        elif lvl == "error": self.sys_logger.error(message)
        else: self.sys_logger.info(message)
