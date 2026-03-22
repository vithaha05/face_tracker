"""
Face Tracker Pipeline Orchestration (Single Source)
--------------------------------------------------
This script integrates detection, recognition, tracking, and logging into a single
real-time pipeline. It processes a single video source (file or RTSP) to identify
individuals, track movement, and log events with unique visitor counting.
"""

import cv2
import json
import logging
import os
import sys
import argparse
from datetime import datetime
import numpy as np

# Import custom modules from the modules folder
from modules.detector import FaceDetector
from modules.recognizer import FaceRecognizer
from modules.tracker import FaceTracker
from modules.logger import EventLogger
from modules.visitor_counter import VisitorCounter
from modules import database

def load_config(config_path: str = "config.json") -> dict:
    """Utility to load the central configuration file."""
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"CRITICAL: Failed to load config: {e}")
            sys.exit(1)
    return {}

def main():
    parser = argparse.ArgumentParser(description="Single-Source Face Tracking Pipeline")
    parser.add_argument("--source", type=str, help="Override video source (file path or RTSP URL)")
    args = parser.parse_args()

    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
    logger = logging.getLogger("Main")
    
    config = load_config()
    source = args.source if args.source else config.get("video_source", "data/sample.mp4")
    
    # Block directory/batch processing
    if isinstance(source, str) and os.path.isdir(source):
        print(f"Error: batch processing not supported. Please provide a single video file or RTSP URL.")
        sys.exit(1)

    logger.info(f"Pipeline starting with source: {source}")

    # Module Initialization
    try:
        db = database.Database()
        detector = FaceDetector()
        recognizer = FaceRecognizer()
        tracker = FaceTracker()
        event_logger = EventLogger()
        visitor_counter = VisitorCounter()
        logger.info("All modules initialized successfully.")
    except Exception as e:
        logger.error(f"Module initialization failed: {e}")
        sys.exit(1)

    # Video Capture Setup
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Stream opened: {width}x{height} @ {fps}fps")

    # Pipeline State
    frame_number = 0
    last_detections = []
    last_known_crops = {} 
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream or failed to read frame.")
                break

            try:
                # ── Step 1: Detection (conditional) ──
                if detector.should_detect(frame_number):
                    detections = detector.detect_faces(frame)
                    last_detections = detections
                else:
                    detections = last_detections

                # ── Step 2: Recognition ──
                embeddings = []
                face_ids = []
                for det in detections:
                    face_crop = detector.crop_face(frame, det["bbox"])
                    face_id = recognizer.identify_or_register(face_crop)
                    if face_id:
                        event_logger.log_embedding_generated(face_id)
                        emb = recognizer.get_embedding(face_crop)
                        embeddings.append(emb)
                        face_ids.append(face_id)
                    else:
                        embeddings.append(None); face_ids.append(None)

                # Filter valid embeddings for tracker
                valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
                tracker_detections = [detections[i] for i in valid_indices]
                tracker_embeddings = [embeddings[i] for i in valid_indices]
                tracker_face_ids = [face_ids[i] for i in valid_indices]

                # ── Step 3: Tracking ──
                active_tracks = tracker.update(tracker_detections, tracker_embeddings, frame, tracker_face_ids)
                
                current_tracker_ids = set()
                for track in active_tracks:
                    tracker_id = track["tracker_id"]
                    current_tracker_ids.add(tracker_id)
                    
                    if track["face_id"]:
                        face_id = track["face_id"]
                        track_crop = detector.crop_face(frame, track["bbox"])
                        last_known_crops[face_id] = track_crop
                        
                        if not hasattr(tracker, '_logged_entries'): tracker._logged_entries = set()
                        if tracker_id not in tracker._logged_entries:
                            image_path = event_logger.log_entry(face_id, track_crop)
                            db.insert_event(face_id, "entry", image_path)
                            visitor_counter.register_entry(face_id)
                            tracker._logged_entries.add(tracker_id)

                # ── Step 4: Exit events ──
                exited_face_ids = tracker.check_exits(current_tracker_ids)
                for face_id in exited_face_ids:
                    last_crop = last_known_crops.get(face_id)
                    image_path = event_logger.log_exit(face_id, last_crop)
                    db.insert_event(face_id, "exit", image_path)
                    logger.info(f"Exit processed for face_id={face_id}")
                    if face_id in last_known_crops: del last_known_crops[face_id]

                # ── Step 5: Visitor count print ──
                if visitor_counter.should_print(frame_number):
                    visitor_counter.print_count()

                # ── Step 6: Visualization ──
                for track in active_tracks:
                    x1, y1, x2, y2 = track["bbox"]
                    face_id = track["face_id"] or "Unknown"
                    conf = track.get("confidence", 0.0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {face_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"Conf: {conf:.2f}", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                cv2.putText(frame, f"Unique Visitors: {visitor_counter.unique_count}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                if config.get("display_output", True):
                    cv2.imshow("Face Tracker", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Quit key pressed. Shutting down.")
                        break

            except Exception as e:
                logger.error(f"Error processing frame {frame_number}: {e}")

            frame_number += 1

    except KeyboardInterrupt:
        logger.info("Interrupted. Shutting down.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        db.close()
        summary = visitor_counter.get_summary()
        logger.info(f"Pipeline stopped. Final summary: {summary}")
        
        print("\n── Session Complete ──")
        print(f"Unique Visitors : {summary['unique_visitors']}")
        print(f"Face IDs Seen   : {len(summary['counted_face_ids'])}")
        print(f"Ended at        : {summary['last_updated']}")
        print("──────────────────────\n")

if __name__ == "__main__":
    main()
