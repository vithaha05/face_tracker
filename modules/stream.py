"""
stream.py — Threaded Video Capture with Auto-Reconnect

Provides a drop-in replacement for cv2.VideoCapture that:
  1. Reads frames in a background thread (eliminates RTSP buffer lag)
  2. Auto-reconnects on stream failure (with configurable retries)
  3. Exposes video metadata (fps, width, height, total_frames) cleanly

Works with:
  - Local video files (*.mp4, *.avi, etc.)
  - Webcam index (0, 1, 2, ...)
  - RTSP URLs (rtsp://...)
  - HTTP/MJPEG streams (http://...)
"""

import cv2
import time
import logging
import threading
from typing import Optional, Tuple, Union

logger = logging.getLogger("VideoStream")


def is_live_source(source: Union[str, int]) -> bool:
    """Determine if the source is a live stream (webcam/RTSP) vs a file."""
    if isinstance(source, int):
        return True
    source_lower = str(source).lower()
    return (
        source_lower.startswith("rtsp://")
        or source_lower.startswith("rtsps://")
        or source_lower.startswith("http://")
        or source_lower.startswith("https://")
        or source_lower == "0"
        or source_lower.isdigit()
    )


class VideoStream:
    """
    Thread-safe video stream reader with auto-reconnect.

    Usage:
        stream = VideoStream(source="rtsp://...", reconnect_attempts=5)
        stream.start()

        while stream.is_running():
            frame = stream.read()
            if frame is None:
                continue
            # process frame...

        stream.stop()
    """

    def __init__(
        self,
        source: Union[str, int] = 0,
        reconnect_attempts: int = 10,
        reconnect_delay: float = 2.0,
        stream_timeout: int = 5000,
    ):
        """
        Initialize the video stream.

        :param source: Video file path, webcam index, or RTSP URL.
        :param reconnect_attempts: Max number of reconnection attempts for live streams.
        :param reconnect_delay: Seconds to wait between reconnection attempts.
        :param stream_timeout: OpenCV stream timeout in milliseconds (for RTSP).
        """
        self.source = source
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.stream_timeout = stream_timeout

        self.is_live = is_live_source(source)

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._consecutive_failures = 0
        self._max_consecutive_failures = 30  # frames

        # Metadata (populated after open)
        self.fps: float = 25.0
        self.width: int = 0
        self.height: int = 0
        self.total_frames: int = 0  # 0 for live streams

    def _open_capture(self) -> bool:
        """Open or re-open the video capture."""
        try:
            if self._cap is not None:
                self._cap.release()

            source = self.source
            if isinstance(source, str) and source.isdigit():
                source = int(source)

            # Use FFMPEG backend for RTSP for better stability
            if isinstance(source, str) and source.lower().startswith("rtsp"):
                self._cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                # Set buffer size to 1 for minimum latency
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                self._cap = cv2.VideoCapture(source)

            if not self._cap.isOpened():
                logger.error(f"Failed to open video source: {source}")
                return False

            # Read metadata
            raw_fps = self._cap.get(cv2.CAP_PROP_FPS)
            self.fps = raw_fps if raw_fps and raw_fps > 0 else 25.0
            self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.total_frames < 0:
                self.total_frames = 0

            logger.info(
                f"Video source opened: {source} | "
                f"{self.width}x{self.height} @ {self.fps:.1f}fps | "
                f"{'LIVE' if self.is_live else f'{self.total_frames} frames'}"
            )
            self._consecutive_failures = 0
            return True

        except Exception as e:
            logger.error(f"Exception opening video source: {e}")
            return False

    def _grab_loop(self):
        """Background thread: continuously grab the latest frame."""
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                if not self._try_reconnect():
                    break
                continue

            ret, frame = self._cap.read()

            if not ret or frame is None:
                self._consecutive_failures += 1

                if self.is_live:
                    if self._consecutive_failures >= self._max_consecutive_failures:
                        logger.warning(
                            f"Lost {self._consecutive_failures} consecutive frames. "
                            f"Attempting reconnect..."
                        )
                        if not self._try_reconnect():
                            break
                    else:
                        time.sleep(0.01)  # Brief pause before retry
                    continue
                else:
                    # File ended
                    logger.info("Video file reached end of stream.")
                    self._running = False
                    break

            # Success — update the shared frame
            self._consecutive_failures = 0
            with self._frame_lock:
                self._frame = frame

    def _try_reconnect(self) -> bool:
        """Attempt to reconnect to the stream."""
        if not self.is_live:
            return False

        for attempt in range(1, self.reconnect_attempts + 1):
            logger.warning(
                f"Reconnect attempt {attempt}/{self.reconnect_attempts} "
                f"to {self.source}..."
            )
            time.sleep(self.reconnect_delay)

            if self._open_capture():
                logger.info(f"Reconnected successfully on attempt {attempt}.")
                self._consecutive_failures = 0
                return True

        logger.error(
            f"Failed to reconnect after {self.reconnect_attempts} attempts. "
            f"Stopping stream."
        )
        self._running = False
        return False

    def start(self) -> bool:
        """Open the stream and start the background reader thread."""
        if not self._open_capture():
            return False

        self._running = True

        if self.is_live:
            # Use threaded capture for live streams to avoid buffer lag
            self._thread = threading.Thread(target=self._grab_loop, daemon=True)
            self._thread.start()
            logger.info("Background frame grabber started (live stream mode).")
        else:
            logger.info("Synchronous capture mode (video file).")

        return True

    def read(self) -> Optional:
        """
        Read the latest frame.

        For live streams: returns the most recent frame from the background thread.
        For files: reads the next frame synchronously (preserving frame order).

        :return: BGR frame (np.ndarray) or None if no frame is available.
        """
        if not self._running:
            return None

        if self.is_live:
            # Return the latest frame grabbed by the background thread
            with self._frame_lock:
                return self._frame.copy() if self._frame is not None else None
        else:
            # Synchronous read for files (preserves frame ordering)
            if self._cap is None:
                return None
            ret, frame = self._cap.read()
            if not ret or frame is None:
                self._running = False
                return None
            return frame

    def is_running(self) -> bool:
        """Check if the stream is still active."""
        return self._running

    def stop(self):
        """Stop the stream and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("Video stream stopped and released.")

    def get_metadata(self) -> dict:
        """Return stream metadata as a dictionary."""
        return {
            "source": str(self.source),
            "is_live": self.is_live,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames,
        }
