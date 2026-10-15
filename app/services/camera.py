import logging
import sys
import threading
import time
from typing import Iterator

import cv2
import numpy as np

from app.services.detection import FaceDetector, annotate

logger = logging.getLogger("face_service.camera")


class CameraManager:
    """Thread-safe wrapper around a single cv2.VideoCapture device."""

    def __init__(self, detector: FaceDetector | None = None):
        self.detector = detector
        self.camera: cv2.VideoCapture | None = None
        self.lock = threading.Lock()
        self.is_running = False

    def start(self, camera_index: int = 0) -> bool:
        with self.lock:
            if self.camera is not None and self.camera.isOpened():
                self.is_running = True
                return True

            # CAP_DSHOW starts much faster on Windows.
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            cap = cv2.VideoCapture(camera_index, backend)
            if not cap.isOpened():
                cap = cv2.VideoCapture(camera_index)

            if cap.isOpened():
                self.camera = cap
                self.is_running = True
                logger.info("Camera opened (index=%s).", camera_index)
                return True

            self.camera = None
            self.is_running = False
            logger.error("Failed to open camera (index=%s).", camera_index)
            return False

    def stop(self) -> bool:
        with self.lock:
            self.is_running = False
            if self.camera is not None:
                try:
                    self.camera.release()
                except Exception as exc:
                    logger.error("Error releasing camera: %s", exc)
                finally:
                    self.camera = None
            logger.info("Camera stopped.")
            return True

    def is_active(self) -> bool:
        with self.lock:
            return self.camera is not None and self.camera.isOpened() and self.is_running

    def capture_frame(self) -> np.ndarray | None:
        with self.lock:
            if self.camera is not None and self.camera.isOpened():
                success, frame = self.camera.read()
                if success and frame is not None:
                    return frame.copy()
            return None

    def gen_frames(self) -> Iterator[bytes]:
        """Yield multipart MJPEG chunks with face boxes and landmarks drawn on each frame."""
        while True:
            with self.lock:
                if not self.is_running or self.camera is None or not self.camera.isOpened():
                    break
                success, frame = self.camera.read()

            if not success or frame is None:
                time.sleep(0.05)
                continue

            if self.detector is not None:
                try:
                    frame = annotate(frame, self.detector.detect(frame), label=False)
                except Exception as exc:
                    logger.warning("Detection error during streaming: %s", exc)

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            time.sleep(0.03)
