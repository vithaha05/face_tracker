import sqlite3
import json
import logging
import time
import os
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Database")

def get_config(config_path: str = "config.json") -> dict:
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

class Database:
    """
    Handles all SQLite database operations for the face tracking system.
    """
    def __init__(self, config_path: str = "config.json"):
        """
        Initializes the database connection and creates tables if they don't exist.
        
        :param config_path: Path to the configuration file to load the db_path from.
        """
        config = get_config(config_path)
        self.db_path = config.get("db_path", "faces_db/faces.db")
        self._initialize_db()

    def _get_connection(self):
        """
        Establishes a connection to the SQLite database and enables WAL mode.
        
        :return: A sqlite3.Connection object.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            return conn
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database at {self.db_path}: {e}")
            raise

    def _initialize_db(self):
        """
        Creates the database directory and tables if they do not exist.
        """
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)

        try:
            with self._get_connection() as conn:
                # Create faces table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS faces (
                        id TEXT PRIMARY KEY,
                        first_seen TIMESTAMP,
                        embedding BLOB
                    )
                """)
                
                # Create events table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        face_id TEXT,
                        event_type TEXT CHECK(event_type IN ('entry', 'exit')),
                        timestamp TIMESTAMP,
                        image_path TEXT,
                        FOREIGN KEY (face_id) REFERENCES faces(id)
                    )
                """)
                conn.commit()
            logger.info("Database initialized successfully.")
        except sqlite3.Error as e:
            logger.error(f"Database initialization failed: {e}")

    def insert_face(self, face_id: str, embedding: np.ndarray) -> bool:
        """
        Inserts a new row into the faces table. Serializes the embedding using tobytes().
        Retries once on failure with a 0.5s delay.
        
        :param face_id: Unique UUID string for the face.
        :param embedding: NumPy array representing the face embedding.
        :return: True on success, False on failure.
        """
        embedding_blob = embedding.tobytes()
        first_seen = datetime.utcnow().isoformat()
        
        for attempt in range(2):
            try:
                with self._get_connection() as conn:
                    conn.execute(
                        "INSERT INTO faces (id, first_seen, embedding) VALUES (?, ?, ?)",
                        (face_id, first_seen, sqlite3.Binary(embedding_blob))
                    )
                    conn.commit()
                return True
            except sqlite3.Error as e:
                logger.warning(f"Attempt {attempt + 1} to insert_face failed: {e}")
                if attempt == 0:
                    time.sleep(0.5)
        
        logger.error(f"Finally failed to insert_face for ID: {face_id}")
        return False

    def insert_event(self, face_id: str, event_type: str, image_path: str) -> bool:
        """
        Inserts a new row into the events table. Auto-fills timestamp with current UTC time.
        Retries once on failure with a 0.5s delay.
        
        :param face_id: Foreign key referencing faces.id.
        :param event_type: Type of event ('entry' or 'exit').
        :param image_path: Path to the image file associated with the event.
        :return: True on success, False on failure.
        """
        timestamp = datetime.utcnow().isoformat()
        
        for attempt in range(2):
            try:
                with self._get_connection() as conn:
                    conn.execute(
                        "INSERT INTO events (face_id, event_type, timestamp, image_path) VALUES (?, ?, ?, ?)",
                        (face_id, event_type, timestamp, image_path)
                    )
                    conn.commit()
                return True
            except sqlite3.Error as e:
                logger.warning(f"Attempt {attempt + 1} to insert_event failed: {e}")
                if attempt == 0:
                    time.sleep(0.5)
        
        logger.error(f"Finally failed to insert_event for face_id: {face_id}")
        return False

    def get_unique_visitor_count(self) -> int:
        """
        Runs a query to count the number of distinct face IDs in the events table.
        
        :return: The integer count of unique visitors.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(DISTINCT face_id) FROM events")
                count = cursor.fetchone()[0]
                return count if count is not None else 0
        except sqlite3.Error as e:
            logger.error(f"Failed to get unique visitor count: {e}")
            return 0

    def get_all_embeddings(self) -> list[dict]:
        """
        Returns all rows from the faces table. Deserializes embeddings using np.frombuffer.
        
        :return: A list of dictionaries containing 'id' and 'embedding' (numpy.ndarray).
        """
        results = []
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, embedding FROM faces")
                rows = cursor.fetchall()
                
                for row in rows:
                    embedding = np.frombuffer(row["embedding"], dtype=np.float32)
                    results.append({
                        "id": row["id"],
                        "embedding": embedding
                    })
            return results
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve embeddings: {e}")
            return []

    def get_all_visitor_ids(self) -> list[str]:
        """
        Fetches all distinct face_ids from the events table where event_type is 'entry'.
        
        :return: A list of unique face ID strings.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT face_id FROM events WHERE event_type = 'entry'")
                rows = cursor.fetchall()
                return [row[0] for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve visitor IDs: {e}")
            return []

    def close(self) -> None:
        """
        Closes the database system.
        """
        logger.info("Database connection system closed.")

if __name__ == "__main__":
    # This block allows for independent testing without affecting imports.
    db = Database()
    print("Database testing complete.")
