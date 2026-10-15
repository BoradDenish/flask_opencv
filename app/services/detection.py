import logging
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.config import Settings

logger = logging.getLogger("face_service.detection")

MAX_DETECT_SIDE = 1280


@dataclass
class Face:
    x: int
    y: int
    w: int
    h: int
    score: float = 1.0
    # YuNet row (x, y, w, h, 5 landmarks, score) in original-image coordinates.
    # Needed by SFace for alignment; None when the Haar fallback produced the box.
    raw: np.ndarray | None = field(default=None, repr=False)

    @property
    def area(self) -> int:
        return self.w * self.h

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


class FaceDetector:
    """YuNet face detector with a Haar cascade fallback when the model file is missing."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = threading.Lock()
        self.yunet = None
        self.haar = None

        if settings.yunet_path.exists():
            self.yunet = cv2.FaceDetectorYN.create(
                str(settings.yunet_path), "", (320, 320), settings.detect_score, 0.3, 5000
            )
            logger.info("Loaded YuNet face detector.")
        else:
            haar_path = settings.haar_dir / "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(str(haar_path))
            if not cascade.empty():
                self.haar = cascade
            logger.warning(
                "YuNet model not found at %s; using Haar cascade (run scripts/download_models.py).",
                settings.yunet_path,
            )

    @property
    def engine(self) -> str:
        if self.yunet is not None:
            return "YuNet"
        if self.haar is not None:
            return "Haar Cascade"
        return "none"

    @property
    def loaded(self) -> bool:
        return self.yunet is not None or self.haar is not None

    def detect(self, img_bgr: np.ndarray) -> list[Face]:
        """Return faces sorted by area, largest first."""
        if self.yunet is not None:
            faces = self.detect_yunet(img_bgr)
        elif self.haar is not None:
            faces = self.detect_haar(img_bgr)
        else:
            return []
        return sorted(faces, key=lambda f: f.area, reverse=True)

    def detect_yunet(self, img_bgr: np.ndarray) -> list[Face]:
        height, width = img_bgr.shape[:2]
        scale = min(1.0, MAX_DETECT_SIDE / max(height, width))
        work = img_bgr if scale == 1.0 else cv2.resize(img_bgr, None, fx=scale, fy=scale)
        with self.lock:
            self.yunet.setInputSize((work.shape[1], work.shape[0]))
            _, rows = self.yunet.detect(work)
        if rows is None:
            return []

        faces = []
        for row in rows:
            row = row.astype(np.float32).copy()
            row[:14] /= scale
            x, y, w, h = (int(round(v)) for v in row[:4])
            # Clamp: YuNet can return boxes that slightly overhang the frame.
            x, y = max(x, 0), max(y, 0)
            w, h = min(w, width - x), min(h, height - y)
            if w <= 0 or h <= 0:
                continue
            faces.append(Face(x, y, w, h, float(row[14]), row))
        return faces

    def detect_haar(self, img_bgr: np.ndarray) -> list[Face]:
        gray = cv2.equalizeHist(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
        with self.lock:
            boxes = self.haar.detectMultiScale(gray, 1.1, 6, minSize=(60, 60))
        return [Face(int(x), int(y), int(w), int(h)) for x, y, w, h in boxes]


def annotate(img_bgr: np.ndarray, faces: list[Face], label: bool = True) -> np.ndarray:
    """Draw boxes, numbering and YuNet landmarks (eyes, nose, mouth) on a copy of the image."""
    out = img_bgr.copy()
    for idx, face in enumerate(faces, start=1):
        cv2.rectangle(out, (face.x, face.y), (face.x + face.w, face.y + face.h), (0, 255, 0), 2)
        if label:
            cv2.putText(
                out, f"Face #{idx}", (face.x, max(20, face.y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
        if face.raw is not None:
            for lx, ly in face.raw[4:14].reshape(5, 2):
                cv2.circle(out, (int(lx), int(ly)), 3, (255, 200, 0), -1)
    return out
