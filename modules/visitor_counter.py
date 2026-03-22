import json
import logging
import os
from datetime import datetime
from typing import Set, List, Dict, Optional
from modules.database import Database

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VisitorCounter")

from modules.utils import load_config

class VisitorCounter:
    """
    Tracks and maintains a count of unique visitors detected by the system.
    """
    def __init__(self, config_path: str = "config.json"):
        """
        Initializes the counter and synchronizes with the existing database entries.
        
        :param config_path: Path to the configuration file.
        """
        self.config = load_config(config_path)
        self.db = Database(config_path)
        
        self.unique_count = 0
        self.counted_faces: Set[str] = set()
        
        # Sync immediately on initialization
        self.sync_from_db()
        logger.info(f"VisitorCounter initialized. Current unique count: {self.unique_count}")

    def sync_from_db(self) -> None:
        """
        Fetches all unique face IDs from existing entry events in the database.
        """
        try:
            face_ids = self.db.get_all_visitor_ids()
            self.counted_faces = set(face_ids)
            self.unique_count = len(self.counted_faces)
            logger.info(f"Synced {self.unique_count} unique visitors from DB.")
        except Exception as e:
            logger.error(f"Failed to sync visitor counter from database: {e}")

    def get_unique_count(self) -> int:
        """
        Returns the current unique count, ensuring it is up-to-date with the DB.
        
        :return: Integer count of unique visitors.
        """
        try:
            # Refresh from total unique IDs in events table
            count = self.db.get_unique_visitor_count()
            self.unique_count = count
            return self.unique_count
        except Exception as e:
            logger.warning(f"Failed to fetch unique count from DB, returning last known: {e}")
            return self.unique_count

    def register_entry(self, face_id: str, image_path: Optional[str] = None) -> bool:
        """
        Records a new entry and increments the counter if the visitor is new.
        
        :param face_id: The identified face ID.
        :param image_path: Optional path to the entry image.
        :return: True if this is a new unique visitor, False otherwise.
        """
        # Always insert an entry event for transparency (or at least track it)
        # However, we only increment the unique count if it's a NEW visitor
        is_new = False
        if face_id not in self.counted_faces:
            self.counted_faces.add(face_id)
            self.unique_count += 1
            is_new = True
            logger.info(f"New unique visitor: face_id={face_id}. Total: {self.unique_count}")
        
        # PERSIST the event to database regardless of whether it's 'new'
        # so that get_unique_visitor_count() (DISTINCT face_id) works!
        self.db.insert_event(face_id, "entry", image_path or "")
        
        return is_new

    def should_print(self, frame_number: int) -> bool:
        """
        Determines if the visitor count should be printed to the console based on interval.
        
        :param frame_number: The current frame index.
        :return: True if it's time to print, False otherwise.
        """
        interval = self.config.get("count_print_interval", 30)
        return frame_number % interval == 0

    def print_count(self) -> None:
        """
        Prints the current unique visitor count to the console and logs it.
        """
        # This is the intentional use of print() for console visibility
        print(f"── Unique Visitors: {self.unique_count} ──")
        logger.debug(f"Console Output: Unique Visitors: {self.unique_count}")

    def get_summary(self) -> dict:
        """
        Returns a summary of the current counter state.
        
        :return: A dictionary containing visitor statistics.
        """
        return {
            "unique_visitors": self.unique_count,
            "counted_face_ids": list(self.counted_faces),
            "last_updated": datetime.utcnow().isoformat()
        }

if __name__ == "__main__":
    # Standalone testing block
    try:
        # Note: Database calls might fail if DB is not properly set up in this environment
        counter = VisitorCounter()
        
        # Test registering new visitors
        counter.register_entry("fake_id_1")
        counter.register_entry("fake_id_2")
        counter.register_entry("fake_id_3")
        
        # Test duplicate entry
        counter.register_entry("fake_id_1")
        
        counter.print_count()
        
        summary = counter.get_summary()
        print("\nSummary Output:")
        print(json.dumps(summary, indent=4))
        
    except Exception as e:
        logger.error(f"Standalone visitor counter test failed: {e}")
