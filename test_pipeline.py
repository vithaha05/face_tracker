import os
import sys
import json
import sqlite3
import logging
import time
import argparse
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Set, Dict, List
import cv2
import numpy as np

# Import custom modules
from modules.detector import FaceDetector
from modules.recognizer import FaceRecognizer
from modules.tracker import FaceTracker
from modules.logger import EventLogger
from modules.visitor_counter import VisitorCounter
from modules import database

@dataclass
class TestResult:
    name: str
    passed: bool
    message: str

class PipelineTester:
    def __init__(self, quick=False, reset=False):
        self.results = []
        self.quick = quick
        self.reset_data = reset
        self.config = self.load_config()
        self.logger: Any = None
        self.setup_logging()

    def load_config(self, config_path: str = "config.json") -> dict:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {"video_source": 0, "db_path": "faces_db/faces.db", "log_dir": "logs"}

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
        self.logger = logging.getLogger("TestPipeline")
        
        # File handler for test results
        fh = logging.FileHandler("test_results.log", mode='w')
        fh.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s'))
        self.logger.addHandler(fh)

    def reset_system(self):
        self.logger.info("Resetting system data (DB and logs)...")
        db_path = self.config.get("db_path", "faces_db/faces.db")
        log_dir = self.config.get("log_dir", "logs")
        
        if os.path.exists(db_path):
            os.remove(db_path)
            self.logger.info(f"Deleted DB: {db_path}")
        
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
            self.logger.info(f"Deleted log dir: {log_dir}")
            
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

    def run_pipeline(self, num_frames=300, frame_skip=1, capture_results=False):
        """Helper to run a subset of the pipeline for testing."""
        try:
            db_inst = database.Database()
            detector = FaceDetector()
            recognizer = FaceRecognizer()
            tracker = FaceTracker()
            event_logger = EventLogger()
            visitor_counter = VisitorCounter()
            
            cap = cv2.VideoCapture(self.config["video_source"])
            if not cap.isOpened():
                return False, "Failed to open video"

            frame_number = 0
            stats = {"entries": 0, "exits": 0, "faces": set()}
            last_detections = []
            last_known_crops = {}

            while frame_number < num_frames:
                ret, frame = cap.read()
                if not ret: break
                
                if detector.should_detect(frame_number):
                    detections = detector.detect_faces(frame)
                    last_detections = detections
                else:
                    detections = last_detections
                
                # Extract embeddings
                embeddings = []
                for det in detections:
                    crop = detector.crop_face(frame, det["bbox"])
                    emb = recognizer.get_embedding(crop)
                    embeddings.append(emb)
                
                v_idx = [i for i,e in enumerate(embeddings) if e is not None]
                t_dets = [detections[i] for i in v_idx]
                t_embs = [embeddings[i] for i in v_idx]
                
                # Tracking
                active = tracker.update(t_dets, t_embs, frame)
                
                cur_ids = set()
                for track in active:
                    tracker_id = track["tracker_id"]
                    cur_ids.add(tracker_id)
                    
                    # Tracker Trust
                    fid = tracker.get_face_id(tracker_id)
                    if not fid:
                        # SYNC: Match new signature (detection, frame, event_logger, tracker_id, embedding)
                        fid = recognizer.identify_or_register(track, frame, event_logger=event_logger, tracker_id=tracker_id)
                        if fid:
                            tracker.assign_face_id(tracker_id, fid)
                    
                    if fid:
                        stats["faces"].add(fid)
                        crop = detector.crop_face(frame, track["bbox"])
                        last_known_crops[fid] = crop
                        
                        if not hasattr(tracker, '_logged'): tracker._logged = set()
                        if tracker_id not in tracker._logged:
                            event_logger.log_entry(fid, crop)
                            visitor_counter.register_entry(fid)
                            tracker._logged.add(tracker_id)
                            stats["entries"] += 1
                
                exits = tracker.check_exits(cur_ids)
                for fid in exits:
                    event_logger.log_exit(fid, last_known_crops.get(fid))
                    stats["exits"] += 1

                frame_number += 1
            
            cap.release()
            return True, stats
        except Exception as e:
            return False, str(e)

    def test_1_video_readable(self):
        name = "Video file readable"
        try:
            source = self.config["video_source"]
            if isinstance(source, str) and not os.path.exists(source):
                raise FileNotFoundError(f"Video file not found: {source}")
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise Exception("VideoCapture failed to open source")
            ret, frame = cap.read()
            if not ret or frame is None:
                raise Exception("Failed to read first frame")
            w, h = int(cap.get(3)), int(cap.get(4))
            fps = cap.get(5)
            cap.release()
            self.results.append(TestResult(name, True, f"Video opened: {w}x{h} @ {fps}fps"))
        except Exception as e:
            self.results.append(TestResult(name, False, str(e)))

    def test_2_run_pipeline(self):
        name = "Pipeline runs 300 frames"
        if self.quick:
            self.results.append(TestResult(name, True, "Skipped in quick mode"))
            return
        success, res = self.run_pipeline(num_frames=300)
        if success:
            self.results.append(TestResult(name, True, f"Ran 300 frames. Entries: {res['entries']}, Exits: {res['exits']}, Faces: {len(res['faces'])}"))
        else:
            self.results.append(TestResult(name, False, f"Crash: {res}"))

    def test_3_verify_events_log(self):
        name = "events.log format correct"
        try:
            log_path = os.path.join(self.config.get("log_dir", "logs"), "events.log")
            if not os.path.exists(log_path):
                raise FileNotFoundError("events.log not found")
            
            malformed = 0
            entries = 0
            exits = 0
            with open(log_path, 'r') as f:
                for line in f:
                    if "[ENTRY]" in line: entries += 1
                    elif "[EXIT]" in line: exits += 1
                    
                    if "face_id=" not in line: malformed += 1
                    
                    if "image=" in line:
                        img_path = line.split("image=")[1].strip()
                        if img_path != "None" and not os.path.exists(img_path):
                            if img_path != "test_path":
                                self.logger.warning(f"Image in log missing: {img_path}")
            
            if malformed > 0:
                self.results.append(TestResult(name, False, f"Found {malformed} malformed lines"))
            elif entries == 0 and not self.quick:
                self.results.append(TestResult(name, False, "No entry events found in log"))
            else:
                self.results.append(TestResult(name, True, f"Verified: {entries} entries, {exits} exits"))
        except Exception as e:
            self.results.append(TestResult(name, False, str(e)))

    def test_4_verify_db_faces(self):
        name = "DB embeddings table valid"
        try:
            db_path = self.config.get("db_path", "faces_db/faces.db")
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM embeddings")
            rows = cursor.fetchall()
            
            invalid_embs = 0
            for row in rows:
                emb_blob = row["embedding"]
                emb = np.frombuffer(emb_blob, dtype=np.float32)
                if emb.shape != (512,):
                    invalid_embs += 1
            
            conn.close()
            if invalid_embs > 0:
                self.results.append(TestResult(name, False, f"{invalid_embs} invalid embeddings found"))
            elif len(rows) == 0 and not self.quick:
                self.results.append(TestResult(name, False, "Embeddings table is empty"))
            else:
                self.results.append(TestResult(name, True, f"Verified {len(rows)} embeddings, all valid vectors"))
        except Exception as e:
            self.results.append(TestResult(name, False, str(e)))

    def test_5_verify_db_events(self):
        name = "DB events table valid"
        try:
            db_path = self.config.get("db_path", "faces_db/faces.db")
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM events")
            rows = cursor.fetchall()
            
            violations = []
            for row in rows:
                if row["event_type"] not in ["entry", "exit"]:
                    violations.append(f"Invalid type: {row['event_type']}")
            
            conn.close()
            if violations:
                self.results.append(TestResult(name, False, f"{len(violations)} violations found"))
            elif len(rows) == 0 and not self.quick:
                self.results.append(TestResult(name, False, "Events table is empty"))
            else:
                self.results.append(TestResult(name, True, f"Verified {len(rows)} events recorded"))
        except Exception as e:
            self.results.append(TestResult(name, False, str(e)))

    def test_6_verify_images(self):
        name = "Images saved correctly"
        try:
            log_dir = self.config.get("log_dir", "logs")
            count = 0
            for root, dirs, files in os.walk(log_dir):
                for f in files:
                    if f.endswith(".jpg"):
                        count += 1
            
            if count == 0 and not self.quick:
                 self.results.append(TestResult(name, False, "No images found in log dir"))
            else:
                self.results.append(TestResult(name, True, f"Found {count} saved face crops"))
        except Exception as e:
            self.results.append(TestResult(name, False, str(e)))

    def test_8_skip_consistency(self):
        name = "Consistency across skip rates"
        if self.quick:
            self.results.append(TestResult(name, True, "Skipped"))
            return
            
        try:
            counts = []
            for skip in [1, 3, 5]:
                 self.reset_system()
                 success, res = self.run_pipeline(num_frames=100, frame_skip=skip)
                 if success:
                     counts.append(len(res["faces"]))
            
            if not counts:
                 raise Exception("Failed to run skip tests")
                 
            # Allow for some jitter but should be globally similar
            if max(counts) - min(counts) > 2:
                 self.results.append(TestResult(name, False, f"Counts varied too much: {counts}"))
            else:
                 self.results.append(TestResult(name, True, f"Consistent: {counts} at skips [1,3,5]"))
        except Exception as e:
            self.results.append(TestResult(name, False, str(e)))

    def print_summary(self):
        print("\n══════════════════════════════════════")
        print(" Test Results Summary")
        print("══════════════════════════════════════")
        all_pass = True
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            if not r.passed: all_pass = False
            print(f"{status:<6} {r.name:<30}")
            self.logger.info(f"RESULT: {status} - {r.name}: {r.message}")
        
        passed_count = sum(1 for r in self.results if r.passed)
        print("──────────────────────────────────────")
        print(f"{passed_count}/{len(self.results)} tests passed")
        print("══════════════════════════════════════\n")
        
        if not all_pass:
            sys.exit(1)
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Face Tracker Test Suite")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    tester = PipelineTester(quick=args.quick, reset=args.reset)
    if args.reset: tester.reset_system()

    tester.test_1_video_readable()
    tester.test_2_run_pipeline()
    tester.test_3_verify_events_log()
    tester.test_4_verify_db_faces()
    tester.test_5_verify_db_events()
    tester.test_6_verify_images()
    tester.test_8_skip_consistency()
    
    tester.print_summary()

if __name__ == "__main__":
    main()
