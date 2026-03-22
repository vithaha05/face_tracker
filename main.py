import cv2
import json
import logging
import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add the project root to sys.path
sys.path.append(str(Path(__file__).parent))

from modules.detector import FaceDetector
from modules.recognizer import FaceRecognizer
from modules.tracker import FaceTracker
from modules.logger import EventLogger
from modules.visitor_counter import VisitorCounter
from modules.database import Database
from modules.utils import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Main")

def draw_overlays(frame: cv2.Mat, active_tracks: List[Dict[str, Any]], unique_count: int, frame_number: int, pending_count: int):
    # Stats Panel
    panel_stats = [
        f"Frame     : {frame_number}",
        f"Tracked   : {len(active_tracks)}",
        f"Unique    : {unique_count}",
        f"Pending   : {pending_count}",
    ]
    y_offset = 30
    for line in panel_stats:
        cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        y_offset += 22

    for track_info in active_tracks:
        face_id = track_info.get("face_id")
        x1, y1, x2, y2 = track_info["bbox"]
        
        color = (0, 255, 0) if face_id else (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID: {face_id if face_id else 'Tracking...'}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def run_frame_pipeline(frame, frame_number, detector, recognizer, tracker, event_logger, visitor_counter, config):
    # Diagnostic counters
    num_embeddings = 0
    num_matches = 0
    num_new_regs = 0
    debug_mode = config.get("debug_mode", False)
    
    # ── Step 2: Detection ──
    detections = detector.detect_all(frame)
    num_raw_faces = len([d for d in detections if "face" in d["type"]])
    num_raw_bodies = len([d for d in detections if "body" in d["type"]])
    
    embeddings = []
    for det in detections:
        emb = None
        # Reuse optimization from previous step
        if det.get("type") == "face" or det.get("face_bbox") is not None:
            bbox = det.get("face_bbox") or det.get("bbox")
            face_crop = detector.crop_face(frame, bbox)
            if face_crop is not None:
                emb = recognizer.get_embedding(face_crop)
        
        det["embedding"] = emb
        embeddings.append(emb)
        if emb is not None: num_embeddings += 1

    # ── Step 3: Tracking ──
    active_tracks = tracker.update(detections, embeddings, frame)
        
    # ── Step 4: Identity Resolution ──
    current_tracker_ids = set()
    for track_info in active_tracks:
        t_id = track_info["tracker_id"]
        current_tracker_ids.add(t_id)
        
        face_id = tracker.get_face_id(t_id)
        if face_id is None:
            best_d = None
            max_iou = 0
            for d in detections:
                iou = detector.compute_iou(d["bbox"], track_info["bbox"])
                if iou > 0.4 and iou > max_iou:
                    max_iou = iou
                    best_d = d
            
            if best_d:
                cached_emb = best_d.get("embedding")
                res_id = recognizer.identify_or_register(best_d, frame, tracker_id=t_id, embedding=cached_emb)
                
                if res_id:
                    tracker.assign_face_id(t_id, res_id)
                    final_bbox = best_d.get("face_bbox") or best_d["bbox"]
                    crop_img = detector.crop_face(frame, final_bbox)
                    crop_img_path = None
                    if crop_img is not None:
                        crop_img_path = event_logger.log_entry(res_id, crop_img)
                    
                    is_new = visitor_counter.register_entry(res_id, crop_img_path)
                    if is_new:
                        num_new_regs += 1
                    else:
                        num_matches += 1
            
    # ── Step 5: Exit Detection ──
    exited_ids = tracker.check_exits(current_tracker_ids)
    for exited_id in exited_ids:
        event_logger.log_exit(exited_id, None)

    if debug_mode:
        msg = f"Frame {frame_number:04} | Raw YOLO (F/B): {num_raw_faces}/{num_raw_bodies} | Embeddings: {num_embeddings} | Matches/Reg: {num_matches}/{num_new_regs} | Total Unique: {visitor_counter.get_unique_count()}"
        print(msg)

    return detections, active_tracks

def main():
    parser = argparse.ArgumentParser(description="High-Performance Face Tracker")
    parser.add_argument("--source", type=str, help="Video file or RTSP URL")
    parser.add_argument("--fast", action="store_true", help="Disable display for maximum speed")
    parser.add_argument("--reset-db", action="store_true", help="Reset DB before running")
    args = parser.parse_args()

    # ── Step 1: Initialization ──
    config = load_config()
    db_path = config.get("db_path", "faces_db/faces.db")
    
    if args.fast:
        config["display_output"] = False
        # Do not override target_process_fps here, use the one from config.json
        config["debug_mode"] = False
        logger.info("FAST MODE enabled: display disabled, debug logs off.")

    if args.reset_db:
        logger.warning("RESET-DB flag detected. Clearing all data...")
        db = Database(db_path)
        db.clear_database()
        import shutil
        if os.path.exists("faces_db/face_images"):
            shutil.rmtree("faces_db/face_images")
        db.close()
        
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    tracker = FaceTracker()
    event_logger = EventLogger()
    visitor_counter = VisitorCounter()
    
    source = args.source if args.source else config.get("video_source", "0")
    cap = cv2.VideoCapture(0 if source == "0" else source)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {source}")
        return

    # Video Properties
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # FIX 1: Auto-calculate frame skip
    target_process_fps = config.get("target_process_fps", 5)
    computed_frame_skip = max(1, int(video_fps / target_process_fps))
    
    # FIX 3: Auto-recalibrate exit_timeout based on computed frame_skip
    # We want ~2 real seconds of absence before exit triggers
    real_seconds_for_exit = config.get("exit_timeout_seconds", 2)
    config["exit_timeout_frames"] = max(5, int(real_seconds_for_exit * target_process_fps))
    
    # FIX 5: Enhanced Startup Log
    logger.info(f"Video: {frame_w}x{frame_h} @ {video_fps}fps | Processing at {config.get('detection_width', 1280)}px width")
    logger.info(f"Frame skip: every {computed_frame_skip} frames | Processing {target_process_fps}fps of {video_fps}fps")
    logger.info(f"Total frames to process: ~{int(total_frames/computed_frame_skip) if total_frames > 0 else 'Stream'}")
    logger.info(f"Exit timeout: {config['exit_timeout_frames']} processed frames ({real_seconds_for_exit}s real time)")

    display_output = config.get("display_output", True)
    
    frame_number = 0
    last_active_tracks = []
    last_detections = []
    start_time = time.time()
    
    try:
        while True:
            # FIX 2: Separate reading from processing
            ret, frame = cap.read()
            if not ret:
                break

            # Only run heavy pipeline every computed_frame_skip frames
            if frame_number % computed_frame_skip == 0:
                last_detections, last_active_tracks = run_frame_pipeline(
                    frame, frame_number, detector, recognizer, tracker, event_logger, visitor_counter, config
                )

            # FIX 4: Progress Bar for files
            if total_frames > 0 and frame_number % (computed_frame_skip * 10) == 0:
                frames_to_process = total_frames // computed_frame_skip
                processed = frame_number // computed_frame_skip
                percent = (frame_number / total_frames) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / max(frame_number, 1)) * (total_frames - frame_number)
                sys.stdout.write(f"\rProgress: {percent:.1f}% | Frame {frame_number}/{total_frames} | "
                      f"Processed: {processed}/{frames_to_process} | "
                      f"Unique: {visitor_counter.get_unique_count()} | "
                      f"ETA: {eta:.0f}s")
                sys.stdout.flush()

            # Step 6: Visualization (Always draw for smoothness)
            if display_output:
                display_frame = frame.copy()
                draw_overlays(display_frame, last_active_tracks, visitor_counter.get_unique_count(), frame_number, len(recognizer.unconfirmed_faces))
                cv2.imshow("Face Tracker", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_number += 1
            
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Fix: Don't pass db_path as config_path
        db = Database()
        db.close()
        
        print("\n\n── Final Diagnostics ──")
        print(f"Total Unique Visitors : {visitor_counter.get_unique_count()}")
        print("─────────────────────")

if __name__ == "__main__":
    main()
