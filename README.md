# Real-Time Face Tracker & Unique Visitor Counter

A production-grade face tracking pipeline designed for counting unique visitors in video streams or live camera feeds. This system integrates multiple AI models and advanced Re-ID strategies to provide robust detection, long-term identity tracking, and accurate counting.

## 🚀 Key Features
- **Accurate Detection**: Optimized YOLOv8-Face (Nano) for high-sensitivity detection.
- **Persistent Re-ID (Fix 1-5)**: 
    - **Multi-Embedding Storage**: Remembers faces from different angles.
    - **Online Profile Updates**: Refines identities in real-time.
    - **Confirmation Buffer**: Prevents ghost tracking and false positives.
    - **Tracker Trust**: Minimizes heavy inference by trusting temporal persistence.
- **Robust Tracking**: DeepSort integration for multi-object tracking across occlusions.
- **Auto-Logging**: Automatic face cropping and event logging for entries and exits.
- **Database Persistence**: SQLite powered for fast, zero-config data storage.
- **Live Camera & RTSP Support**: Threaded capture with auto-reconnect for real-time streams.
- **Graceful Shutdown**: Signal handling ensures all active tracks are flushed as EXIT events.
- **Web Dashboard**: Flask-based read-only dashboard for monitoring tracker output.

---

## 🏗️ System Architecture
The pipeline follows a modular data-flow design:

![System Architecture](assets/architecture.png)
```mermaid
graph TD
    A[Video Source: MP4/RTSP/Webcam] --> B[VideoStream: Threaded Capture]
    B --> C[FaceDetector: YOLOv8-Face + YOLOv8-Body]
    C --> D[FaceRecognizer: InsightFace buffalo_l]
    D --> E[FaceTracker: DeepSort]
    E --> F[Entry/Exit Logic]
    F --> G[SQLite Database]
    F --> H[Structured Log Files]
    F --> I[VisitorCounter]
    I --> J[Live Console / Dashboard]
```

---

## 📂 Project Structure
```
face_tracker/
├── main.py                # Entry point — processes video/stream
├── config.json            # All tunable parameters
├── dashboard.py           # Flask web dashboard (read-only)
├── test_pipeline.py       # Automated test suite
├── requirements.txt       # Python dependencies
├── AI_PLANNING.md         # AI strategy & compute estimates
│
├── modules/
│   ├── __init__.py
│   ├── stream.py          # Threaded video capture with auto-reconnect
│   ├── detector.py        # YOLOv8-Face + Body detection & matching
│   ├── recognizer.py      # InsightFace embedding & identity management
│   ├── tracker.py         # DeepSort tracking & entry/exit logic
│   ├── logger.py          # Event logging (file + image)
│   ├── visitor_counter.py # Unique visitor counting & DB sync
│   ├── database.py        # SQLite operations
│   └── utils.py           # Config loader
│
├── data/                  # Video files (gitignored)
├── faces_db/              # SQLite DB + registered face images
├── logs/                  # Event logs + cropped face images
│   ├── events.log
│   ├── entries/YYYY-MM-DD/
│   └── exits/YYYY-MM-DD/
└── assets/                # Architecture diagram
```

---

## 🛠️ Setup Instructions

### 1. Prerequisites
- Python 3.9+
- macOS (tested on Intel/Silicon) or Linux/Windows
- Git installed

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/vithaha05/face_tracker
cd face_tracker

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download YOLO Models
The YOLO models (`yolov8n-face.pt` and `yolov8n.pt`) are required but gitignored due to size. They will be auto-downloaded by Ultralytics on first run, or you can download them manually.

---

## ▶️ Usage

### Run on a Video File (Development)
```bash
# Default: uses data/sample.mp4 from config.json
python3 main.py

# Custom video file
python3 main.py --source path/to/video.mp4

# Fast mode (no display, maximum processing speed)
python3 main.py --fast
```

### Run on a Live Webcam
```bash
# Default webcam (index 0)
python3 main.py --source 0

# Specific webcam
python3 main.py --source 1
```

### Run on an RTSP Stream (Interview)
```bash
# RTSP stream from an IP camera
python3 main.py --source "rtsp://username:password@192.168.1.100:554/stream1"

# RTSP with display enabled (default)
python3 main.py --source "rtsp://192.168.1.100:554/live"

# RTSP in fast mode (no display window)
python3 main.py --source "rtsp://192.168.1.100:554/live" --fast
```

### Reset Database
```bash
python3 main.py --reset-db
```

### Run Test Suite
```bash
python3 test_pipeline.py --reset
```

### RTSP Stream Configuration
You can also set the RTSP URL in `config.json` instead of passing it via CLI:
```json
{
  "video_source": "rtsp://192.168.1.100:554/stream1"
}
```

---

## 📋 Assumptions

The following assumptions define the operating conditions for which this pipeline was designed and optimised:

| # | Assumption | Details |
|---|------------|---------|
| 1 | **Single video source at a time** | One video file or one RTSP stream per session. Batch processing across multiple simultaneous sources is not supported. |
| 2 | **Stationary camera** | A fixed camera angle is assumed. A moving camera would cause every person to appear as a new entry on each frame. |
| 3 | **Faces at least partially visible** | At least ~50% face visibility is required for InsightFace to generate a reliable embedding. Fully occluded or rear-facing heads may not be detected. |
| 4 | **One person per unique face ID** | Each unique embedding is assumed to belong to one individual. Identical twins or people wearing identical masks may share a face ID. |
| 5 | **Adequate lighting** | Sufficient illumination assumed for YOLO to detect faces at a confidence threshold ≥ 0.25 (configurable via `detection_confidence` in `config.json`). |
| 6 | **No re-entry within exit timeout window** | A person who exits and re-enters within the `exit_timeout_frames` window (default 10 frames) may be treated as a continuous presence rather than a new entry. |
| 7 | **GPU is optional** | The system runs fully on CPU. If a CUDA-enabled GPU is available, InsightFace will automatically use it via `CUDAExecutionProvider`, significantly improving throughput. |
| 8 | **Network stability for RTSP** | For RTSP streams, the system auto-reconnects up to 10 times (configurable) on connection loss. Persistent network failure will stop the stream. |

---

## ⚙️ Configuration (`config.json`)

### Core Parameters
```json
{
  "video_source": "data/sample.mp4",
  "target_process_fps": 5,
  "detection_width": 1280,
  "embedding_confirmation_frames": 1,
  "exit_timeout_seconds": 2,
  "similarity_threshold": 0.35,
  "detection_confidence": 0.25,
  "face_detection_confidence": 0.25,
  "body_detection_confidence": 0.25,
  "display_output": true,
  "debug_mode": false,
  "db_path": "faces_db/faces.db",
  "log_dir": "logs",
  "track_n_init": 1
}
```

### Stream / RTSP Parameters
```json
{
  "reconnect_attempts": 10,
  "reconnect_delay_seconds": 2.0,
  "stream_timeout_ms": 5000,
  "live_display_fps": 30,
  "flush_exits_on_stop": true
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_source` | string | `"data/sample.mp4"` | Default video source (file, webcam index, or RTSP URL) |
| `target_process_fps` | int | `5` | How many frames per second to run the heavy detection pipeline |
| `similarity_threshold` | float | `0.35` | Cosine similarity threshold for face matching |
| `exit_timeout_seconds` | float | `2` | Real-time seconds of absence before a face is considered "exited" |
| `reconnect_attempts` | int | `10` | Max reconnection attempts for live streams |
| `reconnect_delay_seconds` | float | `2.0` | Seconds between reconnection attempts |
| `flush_exits_on_stop` | bool | `true` | Whether to log EXIT events for all active tracks when the system stops |
| `display_output` | bool | `true` | Whether to show the OpenCV display window |

---

## 📐 AI Planning & Compute Load Estimates

### AI Planning Strategy
1. **Hybrid Inference**: Detection and Recognition run only every Nth frame (configurable) to save power, while the **Kalman Filter** (Tracker) runs on every frame to maintain smooth trajectories.
2. **Online Learning**: The system continuously refines a person's average embedding using a moving average, allowing for adaptation to changing lighting or partial occlusions.
3. **Multi-Template Matching**: We store up to 10 varied embeddings per person, checking all of them to find the best match score, significantly reducing "ID flips."
4. **Threaded Capture**: For live streams, a background thread continuously grabs the latest frame, eliminating RTSP buffer lag.

### Compute Load (Estimated for MacBook Air M1/M2)
| Module | CPU Load | Latency (ms) | Notes |
| :--- | :--- | :--- | :--- |
| **YOLO-Face (Detect)** | ~35% | ~25ms | Runs once per N frames |
| **InsightFace (Recognize)** | ~60% | ~80ms per face | Linear with number of faces |
| **DeepSort (Track)** | ~10% | ~5ms | Time-stable overhead |
| **Total System** | ~75% Avg | 10-15 FPS | Optimized for real-time walk-bys |

---

## 🎬 Project Demo

Watch the technical walk-through and demo here:

- 🎥 **Loom Video**: [Watch on Loom](https://www.loom.com/share/0ef4d2b3e2004c9c8a03e88cbd1b3577)
- 📁 **Google Drive**: [View Demo Folder](https://drive.google.com/drive/folders/1O0c1AH0x-7u7DF_SdzoQK2M1Tdqb2VAH?usp=sharing)

---

## 🖥️ Web Dashboard

A read-only Flask web dashboard is included for inspecting tracker output without touching any of the core modules.
```bash
source .venv/bin/activate
pip install flask
python3 dashboard.py
# Open http://localhost:5050
```

| Page | URL | Description |
|------|-----|-------------|
| **Live Dashboard** | `/` | Unique visitor count, today's entries/exits, last 20 events with face thumbnails. Auto-refreshes every 5 seconds. |
| **Face Gallery** | `/faces` | One card per registered face with first-seen timestamp, entry count, and crop image. |
| **Events Log** | `/events` | Paginated (20/page) full event log with Entry/Exit filter buttons. |

> The dashboard works gracefully with an empty database — all pages show "No data yet" instead of crashing.

---

## 🔌 Live Camera / RTSP Quick Reference

### Supported Source Types
| Source | Example | Notes |
|--------|---------|-------|
| **Video file** | `data/sample.mp4` | Processes at max speed |
| **Webcam** | `0` or `1` | Threaded capture, auto-reconnect |
| **RTSP** | `rtsp://192.168.1.100:554/stream1` | FFMPEG backend, 1-frame buffer, auto-reconnect |
| **HTTP/MJPEG** | `http://192.168.1.100:8080/video` | Threaded capture, auto-reconnect |

### Live Stream Behavior
- **Background thread** continuously grabs the latest frame (no buffer lag)
- **Auto-reconnect** on connection drop (configurable retries + delay)
- **Uptime display** on the video overlay for live streams
- **Graceful shutdown** via Ctrl+C flushes all active tracks as EXIT events
- **Periodic stats** logged every 50 processed frames

### Interview Demo Checklist
1. Ensure the RTSP camera is on the same network
2. Set `"video_source": "rtsp://..."` in `config.json` or pass via `--source`
3. Set `"display_output": true` to see the live tracking window
4. Run: `python3 main.py --source "rtsp://IP:PORT/stream"`
5. Press 'q' in the display window or Ctrl+C to stop gracefully

---

### 📝 Hackathon Submission Details
This project is a part of a hackathon run by [Katomaran](https://katomaran.com).

**Submitted before 12 PM Monday March 23rd.**