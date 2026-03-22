import json
import logging
import os
import numpy as np
from typing import Optional, Dict, List, Set, Any
from deep_sort_realtime.deepsort_tracker import DeepSort

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceTracker")

from modules.utils import load_config

class FaceTracker:
    """
    A class for managing face tracking using DeepSort and mapping tracks to face identities.
    """
    def __init__(self, config_path: str = "config.json"):
        """
        Initializes the FaceTracker with DeepSort and internal state management.
        
        :param config_path: Path to the configuration file.
        """
        self.config_path = config_path
        config = load_config(config_path)
        
        max_age = config.get("max_track_age", 30)
        n_init = config.get("track_n_init", 3)
        self.exit_timeout_frames = config.get("exit_timeout_frames", 15)
        
        # Initialize DeepSort
        # embedder=None as we provide our own embeddings from InsightFace
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nn_budget=100,
            embedder=None
        )
        
        # Internal state
        self.track_to_face: Dict[int, str] = {}    # tracker_id -> face_id
        self.lost_tracks: Dict[int, int] = {}       # tracker_id -> frames_missing
        self.active_tracks: Set[int] = set()       # set of currently confirmed tracker_ids
        
        logger.info(f"FaceTracker initialized (max_age={max_age}, n_init={n_init}, exit_timeout={self.exit_timeout_frames}).")

    def update(self, detections: List[Dict[str, Any]], embeddings: List[np.ndarray], frame: np.ndarray, face_ids: Optional[List[Optional[str]]] = None) -> List[Dict[str, Any]]:
        """
        Updates the tracker with new detections and embeddings for the current frame.
        
        :param detections: List of detections from detector.py.
        :param embeddings: List of corresponding embeddings from recognizer.py.
        :param frame: The current BGR frame.
        :param face_ids: Optional list of identified face IDs for each detection.
        :return: A list of active confirmed track dictionaries.
        """
        if frame is None:
            return []

        # Convert detections to DeepSort format
        ds_input = []
        ds_embeds = []
        for i, (det, emb) in enumerate(zip(detections, embeddings)):
            bbox = det["bbox"]
            left, top, right, bottom = bbox
            width = right - left
            height = bottom - top
            # Use provided face_id as label if available, otherwise "face"
            label = face_ids[i] if (face_ids is not None and face_ids[i]) else "face"
            
            # Fix: Ensure embedding is NOT None and valid for DeepSort
            if emb is None:
                # Use a small constant vector and normalize it to avoid div-by-zero
                emb = np.ones(512, dtype=np.float32) * 0.01
                emb = emb / np.linalg.norm(emb)
                
            ds_input.append(([left, top, width, height], det["confidence"], label))
            ds_embeds.append(emb)
            
        active_confirmed_tracks = []
        
        try:
            # Update tracks with explicit embeds
            tracks = self.tracker.update_tracks(ds_input, embeds=ds_embeds, frame=frame)
            
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                tracker_id = int(track.track_id)
                self.active_tracks.add(tracker_id)
                
                # Retrieve face_id from track label if it's a valid ID (not "face")
                current_face_id = track.det_class
                if current_face_id and current_face_id != "face":
                    # Optionally update the mapping if we got a new ID
                    self.track_to_face[tracker_id] = current_face_id
                
                # Remove from lost tracks if it was missing
                if tracker_id in self.lost_tracks:
                    del self.lost_tracks[tracker_id]
                
                # Get bbox in [x1, y1, x2, y2] format
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                
                active_confirmed_tracks.append({
                    "tracker_id": tracker_id,
                    "face_id": self.track_to_face.get(tracker_id),
                    "bbox": [x1, y1, x2, y2],
                    "confidence": track.get_det_conf() or 1.0 # Use detection confidence
                })
                
            return active_confirmed_tracks
            
        except Exception as e:
            logger.error(f"Error during tracker update: {e}")
            return []

    def assign_face_id(self, tracker_id: int, face_id: str) -> None:
        """
        Maps a tracker ID to a identified/registered face ID and logs entry events.
        
        :param tracker_id: The ID assigned by DeepSort.
        :param face_id: The unique identity string from the recognizer.
        """
        if tracker_id not in self.track_to_face:
            logger.info(f"Entry event: face_id={face_id}, tracker_id={tracker_id}")
            
        self.track_to_face[tracker_id] = face_id

    def check_exits(self, current_tracker_ids: Set[int]) -> List[str]:
        """
        Checks for confirmed tracks that have disappeared and triggers exit events after timeout.
        
        :param current_tracker_ids: The set of tracker IDs visible in the current frame.
        :return: A list of face IDs that have officially exited.
        """
        exited_face_ids = []
        tracks_to_remove = []
        
        # Check all tracks we previously considered active
        for tracker_id in list(self.active_tracks):
            if tracker_id not in current_tracker_ids:
                # Track is missing in this frame
                self.lost_tracks[tracker_id] = self.lost_tracks.get(tracker_id, 0) + 1
                
                if self.lost_tracks[tracker_id] >= self.exit_timeout_frames:
                    face_id = self.track_to_face.get(tracker_id)
                    if face_id:
                        exited_face_ids.append(face_id)
                        logger.info(f"Exit event: face_id={face_id}, tracker_id={tracker_id}")
                    
                    tracks_to_remove.append(tracker_id)
            else:
                # Track is present, ensure it's removed from lost if it was there
                if tracker_id in self.lost_tracks:
                    del self.lost_tracks[tracker_id]
        
        # Cleanup internal state
        for tracker_id in tracks_to_remove:
            self.active_tracks.remove(tracker_id)
            if tracker_id in self.lost_tracks:
                del self.lost_tracks[tracker_id]
            if tracker_id in self.track_to_face:
                del self.track_to_face[tracker_id]
                
        return exited_face_ids

    def get_face_id(self, tracker_id: int) -> Optional[str]:
        """
        Returns the face ID associated with a tracker ID, if any.
        
        :param tracker_id: The tracker ID to look up.
        :return: The associated face ID or None.
        """
        return self.track_to_face.get(tracker_id)

    def reset(self) -> None:
        """
        Resets the internal state and the DeepSort tracker.
        """
        config = load_config(self.config_path)
        max_age = config.get("max_track_age", 30)
        n_init = config.get("track_n_init", 3)
        
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nn_budget=100,
            embedder=None
        )
        self.track_to_face = {}
        self.lost_tracks = {}
        self.active_tracks = set()
        logger.info("FaceTracker internal state and DeepSort reset.")

if __name__ == "__main__":
    # Standalone testing block
    try:
        tracker = FaceTracker()
        
        # Simulate 5 frames of detections
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        for frame_idx in range(1, 6):
            # Fake detection: [left, top, right, bottom]
            # Moving detection slightly each frame
            x1, y1 = 100 + frame_idx * 5, 100
            x2, y2 = x1 + 50, y1 + 50
            
            detections = [{"bbox": [x1, y1, x2, y2], "confidence": 0.95}]
            embeddings = [np.random.rand(512).astype(np.float32)]
            
            active_tracks = tracker.update(detections, embeddings, dummy_frame)
            
            current_ids = {t["tracker_id"] for t in active_tracks}
            print(f"Frame {frame_idx}: Confirmed tracks count = {len(active_tracks)}")
            
            for track in active_tracks:
                tracker.assign_face_id(track["tracker_id"], "test_face_id_1")
                print(f"  Track ID {track['tracker_id']} -> face_id {track['face_id']}")
            
            exits = tracker.check_exits(current_ids)
            if exits:
                print(f"  Exits detected: {exits}")
                
        # Simulate disappearance
        print("\nSimulating disappearance for 20 frames...")
        for frame_idx in range(6, 26):
            # No detections
            active_tracks = tracker.update([], [], dummy_frame)
            current_ids = {t["tracker_id"] for t in active_tracks}
            exits = tracker.check_exits(current_ids)
            if exits:
                print(f"Frame {frame_idx}: Face(s) {exits} have exited after timeout.")
                break
                
    except Exception as e:
        logger.error(f"Standalone tracker test failed: {e}")
