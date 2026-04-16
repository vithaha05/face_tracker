import cv2
import json
import logging
import os
import sys
import signal
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
from modules.stream import VideoStream, is_live_source
from modules.utils import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Main")

# ── Global flag for graceful shutdown ──
_shutdown_requested = False

def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    global _shutdown_requested
    if not _shutdown_requested:
        logger.info(f"Shutdown signal received (signal {signum}). Finishing current frame...")
        _shutdown_requested = True
    else:
        logger.warning("Force quit requested. Exiting immediately.")
        sys.exit(1)

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def draw_overlays(frame: cv2.Mat, active_tracks: List[Dict[str, Any]], unique_count: int, frame_number: int, pending_count: int, is_live: bool = False, uptime: float = 0.0):
    """Draw tracking info overlays on the frame."""
    # Stats Panel
    panel_stats = [
        f"Frame     : {frame_number}",
        f"Tracked   : {len(active_tracks)}",
        f"Unique    : {unique_count}",
        f"Pending   : {pending_count}",
    ]
    if is_live:
        mins, secs = divmod(int(uptime), 60)
        hrs, mins = divmod(mins, 60)
        panel_stats.append(f"Uptime    : {hrs:02d}:{mins:02d}:{secs:02d}")

    y_offset = 30
    for line in panel_stats:
        cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        y_offset += 22

    # Draw bounding boxes for tracked faces
    for track_info in active_tracks:
        face_id = track_info.get("face_id")
        x1, y1, x2, y2 = track_info["bbox"]
        
        color = (0, 255, 0) if face_id else (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID: {face_id if face_id else 'Tracking...'}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Source indicator
    source_label = "LIVE" if is_live else "FILE"
    source_color = (0, 0, 255) if is_live else (0, 200, 0)
    h, w = frame.shape[:2]
    cv2.putText(frame, source_label, (w - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, source_color, 2)


def run_frame_pipeline(frame, frame_number, detector, recognizer, tracker, event_logger, visitor_counter, config):
    """Run the full detection → recognition → tracking pipeline on a single frame."""
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
                res_id = recognizer.identify_or_register(best_d, frame, event_logger=event_logger, tracker_id=t_id, embedding=cached_emb)
                
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
        print(f"EXIT: {exited_id}")
        crop_path = event_logger.log_exit(exited_id, None)
        visitor_counter.register_exit(exited_id, crop_path)

    if debug_mode:
        msg = f"Frame {frame_number:04} | Raw YOLO (F/B): {num_raw_faces}/{num_raw_bodies} | Embeddings: {num_embeddings} | Matches/Reg: {num_matches}/{num_new_regs} | Total Unique: {visitor_counter.get_unique_count()}"
        print(msg)

    return detections, active_tracks


def flush_remaining_exits(tracker, event_logger, visitor_counter):
    """
    Flush all currently active tracks as EXIT events.
    Called during shutdown to ensure every entry has a corresponding exit.
    """
    flushed = 0
    for tracker_id in list(tracker.active_tracks):
        face_id = tracker.track_to_face.get(tracker_id)
        if face_id:
            print(f"EXIT: {face_id}")
            crop_path = event_logger.log_exit(face_id, None)
            visitor_counter.register_exit(face_id, crop_path)
            flushed += 1
    
    # Clean up tracker state
    tracker.active_tracks.clear()
    tracker.track_to_face.clear()
    tracker.lost_tracks.clear()
    
    if flushed > 0:
        logger.info(f"Flushed {flushed} remaining active tracks as EXIT events.")
    return flushed


def main():
    global _shutdown_requested

    parser = argparse.ArgumentParser(description="High-Performance Face Tracker")
    parser.add_argument("--source", type=str, help="Video file path, RTSP URL, or webcam index (e.g., 0)")
    parser.add_argument("--fast", action="store_true", help="Disable display for maximum speed")
    parser.add_argument("--reset-db", action="store_true", help="Reset DB before running")
    args = parser.parse_args()

    # ── Step 1: Source & Reset Logic ──
    config = load_config()
    db_path = config.get("db_path", "faces_db/faces.db")
    source = args.source if args.source else config.get("video_source", "0")
    source_str = str(source)
    
    # ── Handle Manual or Auto Reset ──
    last_source_file = ".last_source"
    last_source = ""
    if os.path.exists(last_source_file):
        try:
            with open(last_source_file, "r") as f:
                last_source = f.read().strip()
        except Exception: pass

    should_reset = args.reset_db or (last_source != "" and last_source != source_str)
    
    if should_reset:
        reason = "RESET-DB flag" if args.reset_db else f"Source changed from '{last_source}' to '{source_str}'"
        logger.warning(f"{reason}. Clearing all system data...")
        
        db = Database(db_path)
        db.clear_database()
        import shutil
        if os.path.exists("faces_db/face_images"):
            shutil.rmtree("faces_db/face_images")
        log_dir = config.get("log_dir", "logs")
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
            os.makedirs(log_dir, exist_ok=True)
        db.close()

    # Save current source state
    try:
        with open(last_source_file, "w") as f:
            f.write(source_str)
    except Exception as e:
        logger.error(f"Failed to save state: {e}")

    # ── Step 2: Initialize Modules (after potential reset) ──
    if args.fast:
        config["display_output"] = False
        config["debug_mode"] = False
        logger.info("FAST MODE enabled.")

    detector = FaceDetector()
    recognizer = FaceRecognizer()
    tracker = FaceTracker()
    event_logger = EventLogger()
    visitor_counter = VisitorCounter()

    # ── Step 2: Resolve video source strings for display ──
    if isinstance(source, str) and source.isdigit():
        source_display = f"Webcam {source}"
    elif isinstance(source, str) and source.lower().startswith("rtsp"):
        source_display = f"RTSP Stream"
    else:
        source_display = source

    logger.info(f"Source: {source_display}")

    # ── Open stream using threaded VideoStream ──
    stream = VideoStream(
        source=source,
        reconnect_attempts=config.get("reconnect_attempts", 10),
        reconnect_delay=config.get("reconnect_delay_seconds", 2.0),
        stream_timeout=config.get("stream_timeout_ms", 5000),
    )
    
    if not stream.start():
        logger.error(f"Cannot open video source: {source}")
        return

    # Video Properties (from stream metadata)
    video_fps = stream.fps
    total_frames = stream.total_frames

    # Auto-calculate frame skip based on desired processing FPS
    target_process_fps = config.get("target_process_fps", 5)
    computed_frame_skip = max(1, int(video_fps / target_process_fps))
    
    # Auto-recalibrate exit_timeout based on computed frame_skip
    real_seconds_for_exit = config.get("exit_timeout_seconds", 2)
    config["exit_timeout_frames"] = max(5, int(real_seconds_for_exit * target_process_fps))
    
    # Startup Log
    logger.info(f"Video: {stream.width}x{stream.height} @ {video_fps:.1f}fps | Processing at {config.get('detection_width', 1280)}px width")
    logger.info(f"Frame skip: every {computed_frame_skip} frames | Processing {target_process_fps}fps of {video_fps:.1f}fps")
    
    if stream.is_live:
        logger.info(f"Mode: LIVE STREAM | Reconnect attempts: {config.get('reconnect_attempts', 10)}")
    else:
        logger.info(f"Total frames to process: ~{int(total_frames / computed_frame_skip) if total_frames > 0 else 'Unknown'}")
    
    logger.info(f"Exit timeout: {config['exit_timeout_frames']} processed frames ({real_seconds_for_exit}s real time)")

    display_output = config.get("display_output", True)
    
    frame_number = 0
    last_active_tracks = []
    last_detections = []
    start_time = time.time()
    frames_processed = 0
    last_stats_at = 0  # Track when we last printed live stats
    
    try:
        while stream.is_running() and not _shutdown_requested:
            frame = stream.read()
            
            if frame is None:
                if stream.is_live:
                    # Live stream: brief wait, frame will come from background thread
                    time.sleep(0.001)
                    continue
                else:
                    # File ended
                    break

            # Only run heavy pipeline every computed_frame_skip frames
            if frame_number % computed_frame_skip == 0:
                last_detections, last_active_tracks = run_frame_pipeline(
                    frame, frame_number, detector, recognizer, tracker, event_logger, visitor_counter, config
                )
                frames_processed += 1

            # Progress / Status display
            elapsed = time.time() - start_time
            
            if not stream.is_live:
                # File mode: show progress bar
                if total_frames > 0 and frame_number % (computed_frame_skip * 10) == 0 and frame_number > 0:
                    frames_to_process = total_frames // computed_frame_skip
                    processed = frame_number // computed_frame_skip
                    percent = (frame_number / total_frames) * 100
                    eta = (elapsed / max(frame_number, 1)) * (total_frames - frame_number)
                    sys.stdout.write(f"\rProgress: {percent:.1f}% | Frame {frame_number}/{total_frames} | "
                          f"Processed: {processed}/{frames_to_process} | "
                          f"Unique: {visitor_counter.get_unique_count()} | "
                          f"ETA: {eta:.0f}s")
                    sys.stdout.flush()
            else:
                # Live mode: show periodic status (once per 50 processed frames)
                if frames_processed > 0 and frames_processed % 50 == 0 and frames_processed != last_stats_at:
                    last_stats_at = frames_processed
                    actual_fps = frames_processed / max(elapsed, 0.01)
                    logger.info(
                        f"Live Stats | Uptime: {elapsed:.0f}s | "
                        f"Frames processed: {frames_processed} | "
                        f"Processing FPS: {actual_fps:.1f} | "
                        f"Unique visitors: {visitor_counter.get_unique_count()} | "
                        f"Active tracks: {len(last_active_tracks)}"
                    )

            # Visualization
            if display_output:
                display_frame = frame.copy()
                draw_overlays(
                    display_frame, last_active_tracks, 
                    visitor_counter.get_unique_count(), frame_number, 
                    len(recognizer.unconfirmed_faces),
                    is_live=stream.is_live, uptime=elapsed
                )
                cv2.imshow("Face Tracker", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit key pressed.")
                    break

            frame_number += 1
            
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # ── Graceful Shutdown ──
        logger.info("Shutting down gracefully...")
        
        # Flush all remaining active tracks as EXIT events
        if config.get("flush_exits_on_stop", True):
            flush_remaining_exits(tracker, event_logger, visitor_counter)
        
        # Stop stream
        stream.stop()
        cv2.destroyAllWindows()
        
        # Close database
        db = Database()
        db.close()
        
        # Final diagnostics
        elapsed = time.time() - start_time
        print("\n\n── Final Diagnostics ──")
        print(f"Total Unique Visitors : {visitor_counter.get_unique_count()}")
        print(f"Frames Read           : {frame_number}")
        print(f"Frames Processed      : {frames_processed}")
        print(f"Total Time            : {elapsed:.1f}s")
        if frames_processed > 0:
            print(f"Avg Processing FPS    : {frames_processed / elapsed:.1f}")
        print(f"Source                : {'LIVE' if stream.is_live else 'FILE'} ({source})")
        print("─────────────────────")

if __name__ == "__main__":
    main()
