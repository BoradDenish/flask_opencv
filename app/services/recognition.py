import logging
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import Settings
from app.services.detection import Face

logger = logging.getLogger("face_service.recognition")


@dataclass
class Match:
    user: dict
    score: float
    confidence: float


class FaceRecognizer:
    """SFace embedding extractor with cosine-similarity matching."""

    engine = "OpenCV SFace"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = threading.Lock()
        self.model = None
        if settings.sface_path.exists():
            self.model = cv2.FaceRecognizerSF.create(str(settings.sface_path), "")
            logger.info("Loaded SFace recognizer.")
        else:
            logger.warning(
                "SFace model not found at %s; recognition disabled (run scripts/download_models.py).",
                settings.sface_path,
            )

    @property
    def available(self) -> bool:
        return self.model is not None

    def embed(self, img_bgr: np.ndarray, face: Face) -> np.ndarray:
        """Align the face using its landmarks and return an L2-normalised 128-d embedding."""
        if self.model is None or face.raw is None:
            raise RuntimeError("Face recognition model is not available")
        with self.lock:
            aligned = self.model.alignCrop(img_bgr, face.raw)
            feature = self.model.feature(aligned)
        vec = feature.flatten().astype(np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def confidence(self, score: float) -> float:
        """Map cosine score to 0..1 so the decision threshold sits at 0.5."""
        t = self.settings.match_threshold
        if score >= t:
            return min(1.0, 0.5 + 0.5 * (score - t) / (1.0 - t))
        return max(0.0, 0.5 * score / t)

    def best_match(
        self, query: np.ndarray, users: list[dict], matrix: np.ndarray
    ) -> tuple[Match | None, Match | None]:
        """Return (accepted match, closest candidate). The first is None below the threshold."""
        if len(users) == 0:
            return None, None
        scores = matrix @ query
        idx = int(np.argmax(scores))
        score = float(scores[idx])
        candidate = Match(users[idx], score, self.confidence(score))
        accepted = candidate if score >= self.settings.match_threshold else None
        return accepted, candidate
