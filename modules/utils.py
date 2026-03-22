import json
import logging
import os

logger = logging.getLogger("Utils")

def load_config(config_path: str = "config.json") -> dict:
    """
    Robustly loads the configuration file.
    Returns a default configuration if the file is missing or corrupted.
    """
    default_config = {
        "video_source": "0",
        "db_path": "faces_db/faces.db",
        "log_dir": "logs",
        "similarity_threshold": 0.5,
        "debug_mode": False
    }
    
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found. Using defaults.")
        return default_config
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                logger.error(f"Config file {config_path} is empty. Using defaults.")
                return default_config
            return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse config file {config_path}: {e}. Using defaults.")
        return default_config
    except Exception as e:
        logger.error(f"Error reading config file {config_path}: {e}. Using defaults.")
        return default_config
