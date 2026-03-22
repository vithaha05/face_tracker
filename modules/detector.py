import os
import json
import logging
import cv2
import numpy as np
from typing import List, Dict, Any
from ultralytics import YOLO

# Configure logging to be very visible
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceDetector")

from modules.utils import load_config

class FaceDetector:
    """
    Robust Detector using YOLO-Face and YOLO-Body with permissive filters.
    """
    def __init__(self, config_path: str = "config.json"):
        config = load_config(config_path)
        yolo_face_path = config.get("yolo_model_path", "yolov8n-face.pt")
        yolo_body_path = "yolov8n.pt"

        self.face_conf = config.get("face_detection_confidence", 0.25)
        self.body_conf = config.get("body_detection_confidence", 0.25)
        self.frame_skip = config.get("frame_skip", 1)
        self.face_vis_threshold = config.get("face_visibility_threshold", 0.15)
        self.body_vis_threshold = config.get("body_visibility_threshold", 0.40)
        self.iou_merge_threshold = config.get("iou_merge_threshold", 0.20)
        self.debug_mode = config.get("debug_mode", True)
        self.detection_width = config.get("detection_width", 1280)

        try:
            self.face_model = YOLO(yolo_face_path)
            logger.info(f"YOLO Face model loaded from {yolo_face_path}.")
            self.body_model = YOLO(yolo_body_path)
            logger.info("YOLO Body model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO models: {e}")
            raise

    def should_detect(self, frame_number: int) -> bool:
        return frame_number % self.frame_skip == 0

    def _preprocess_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """Resize frame to target width for detection. Returns resized frame and scale factor."""
        h, w = frame.shape[:2]
        if w <= self.detection_width:
            return frame, 1.0
        scale = self.detection_width / w
        new_h = int(h * scale)
        resized = cv2.resize(frame, (self.detection_width, new_h))
        return resized, scale

    def _run_face_model(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        resized_frame, scale = self._preprocess_frame(frame)
        results = self.face_model(resized_frame, conf=self.face_conf, verbose=False, imgsz=320)
        detections = []
        for r in results:
            for box in r.boxes:
                bbox = [int(v / scale) for v in box.xyxy[0].tolist()]
                detections.append({
                    "bbox": bbox,
                    "confidence": float(box.conf),
                    "type": "face"
                })
        return detections

    def _run_body_model(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        resized_frame, scale = self._preprocess_frame(frame)
        results = self.body_model(resized_frame, conf=self.body_conf, verbose=False, imgsz=320)
        detections = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0: # person
                    bbox = [int(v / scale) for v in box.xyxy[0].tolist()]
                    detections.append({
                        "bbox": bbox,
                        "confidence": float(box.conf),
                        "type": "body"
                    })
        return detections

    def estimate_visibility(self, bbox: list, frame_shape: tuple) -> float:
        h, w = frame_shape[:2]
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        
        # Simple ratio relative to 40% height for body, or 5% height for face
        # We'll just return relative height for body
        return bh / h

    def compute_iou(self, box1: list, box2: list) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = max(1, area1 + area2 - intersection)
        return intersection / union

    def detect_all(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if frame is None: return []
        
        # Step 1: Faces
        faces = self._run_face_model(frame)
        if self.debug_mode: logger.debug(f"Raw face detections: {len(faces)}")
        
        # Step 2: Bodies
        bodies = self._run_body_model(frame)
        if self.debug_mode: logger.debug(f"Raw body detections: {len(bodies)}")
        
        # Step 3: Match using Containment (Face is almost always inside body)
        matched_results = []
        used_faces = set()
        
        for b in bodies:
            b_box = b["bbox"]
            b_vis = self.estimate_visibility(b_box, frame.shape)
            best_face = None
            max_coverage = 0
            face_idx = -1
            
            for i, f in enumerate(faces):
                if i in used_faces: continue
                f_box = f["bbox"]
                
                # Compute Intersection / Face Area (Coverage)
                # This is much more robust for face matching than standard IoU
                x1 = max(b_box[0], f_box[0])
                y1 = max(b_box[1], f_box[1])
                x2 = min(b_box[2], f_box[2])
                y2 = min(b_box[3], f_box[3])
                
                if x2 > x1 and y2 > y1:
                    inter_area = (x2 - x1) * (y2 - y1)
                    face_area = (f_box[2] - f_box[0]) * (f_box[3] - f_box[1])
                    coverage = inter_area / max(1, face_area)
                    
                    if coverage > self.iou_merge_threshold: # Reuse this threshold (permissively)
                        if coverage > max_coverage:
                            max_coverage = coverage
                            best_face = f
                            face_idx = i
            
            det = {
                "bbox": b["bbox"],
                "confidence": b["confidence"],
                "body_vis": b_vis,
                "face_bbox": None,
                "face_vis": 0.0,
                "type": "body_only"
            }
            
            if best_face:
                det["face_bbox"] = best_face["bbox"]
                det["face_vis"] = self.estimate_visibility(best_face["bbox"], frame.shape) # not used but for consistency
                det["type"] = "body+face"
                used_faces.add(face_idx)
                
            matched_results.append(det)
            
        # Add faces that weren't matched to bodies
        for i, f in enumerate(faces):
            if i not in used_faces:
                matched_results.append({
                    "bbox": f["bbox"],
                    "face_bbox": f["bbox"],
                    "confidence": f["confidence"],
                    "body_vis": 0.0,
                    "face_vis": 1.0, # Always visible if detected as face
                    "type": "face_only"
                })
        
        if self.debug_mode: logger.debug(f"After matching: {len(matched_results)}")
        
        # Step 4: Visibility filter LOOSELY
        filtered = []
        for d in matched_results:
            # RULE: If face detected (face_only or body+face) -> keep it always
            if d["face_bbox"] is not None:
                filtered.append(d)
                continue
            
            # RULE: If body only -> check visibility threshold
            if d["body_vis"] >= self.body_vis_threshold:
                filtered.append(d)
            else:
                if self.debug_mode: 
                    logger.debug(f"Dropped detection: body_vis={d['body_vis']:.2f} < threshold={self.body_vis_threshold}, no face present")
        
        if self.debug_mode: logger.debug(f"Final filtered detections: {len(filtered)}")
        return filtered

    def crop_face(self, frame: np.ndarray, bbox: list, padding_ratio=0.4) -> np.ndarray:
        """
        FIX 4: Generous padding for InsightFace.
        """
        if bbox is None:
            return None
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        bw = x2 - x1
        bh = y2 - y1
        pad_x = int(bw * padding_ratio)
        pad_y = int(bh * padding_ratio)
        
        nx1 = max(0, x1 - pad_x)
        ny1 = max(0, y1 - pad_y)
        nx2 = min(w, x2 + pad_x)
        ny2 = min(h, y2 + pad_y)
        
        crop = frame[ny1:ny2, nx1:nx2]
        if crop.shape[0] < 112 or crop.shape[1] < 112:
            crop = cv2.resize(crop, (112, 112))
        return crop
