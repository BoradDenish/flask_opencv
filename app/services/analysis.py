import logging
import os
from typing import Any

import numpy as np

from app.config import get_settings

logger = logging.getLogger("face_service.analysis")

# Keep DeepFace weights inside the project (models/deepface/.deepface/weights)
# instead of ~/.deepface. Must be set before deepface is imported.
os.environ.setdefault("DEEPFACE_HOME", str(get_settings().models_dir / "deepface"))

try:
    from deepface import DeepFace

    DEEPFACE_AVAILABLE = True
except Exception as exc:  # TensorFlow is often missing; analysis then degrades to "unavailable".
    DeepFace = None
    DEEPFACE_AVAILABLE = False
    logger.info("DeepFace not available (%s); age/gender/emotion analysis is disabled.", exc)


def analyze_attributes(img_bgr: np.ndarray, detection_engine: str, faces_detected: int) -> dict[str, Any]:
    """Estimate age, gender and emotion. Values are None when no model can provide them."""
    if not DEEPFACE_AVAILABLE:
        return {
            "age": None,
            "gender": None,
            "emotion": {"dominant_emotion": None, "details": {"faces_detected": faces_detected}},
            "faces_detected": faces_detected,
            "engine": f"{detection_engine} (detection only)",
            "note": "Age, gender and emotion need DeepFace: pip install deepface tensorflow (Python <= 3.12).",
        }

    result = DeepFace.analyze(
        img_path=img_bgr, actions=["age", "gender", "emotion"], enforce_detection=False
    )
    item = result[0] if isinstance(result, list) else result
    gender = item.get("dominant_gender") or item.get("gender")
    if isinstance(gender, dict):
        gender = max(gender, key=gender.get)
    return {
        "age": int(item["age"]) if item.get("age") is not None else None,
        "gender": gender,
        "emotion": {
            "dominant_emotion": item.get("dominant_emotion"),
            "details": {k: float(v) for k, v in (item.get("emotion") or {}).items()},
        },
        "faces_detected": faces_detected,
        "engine": "DeepFace",
    }
