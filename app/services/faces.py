"""High-level face workflows shared by the routers."""
import cv2
import numpy as np

from app import database
from app.config import Settings
from app.errors import ApiError
from app.services.detection import Face, FaceDetector
from app.services.recognition import FaceRecognizer
from app.services.storage import resolve_static_path


def require_single_face(
    img: np.ndarray, detector: FaceDetector, settings: Settings, min_size: int | None = None
) -> Face:
    """Return the only face in the image, or raise a descriptive 400."""
    faces = detector.detect(img)
    if not faces:
        raise ApiError(400, "No face detected in the photo. Please use a clear photo with a visible face.")
    if len(faces) > 1:
        raise ApiError(400, f"{len(faces)} faces detected. Please use a photo with exactly one face.")
    minimum = settings.min_face_size if min_size is None else min_size
    if faces[0].w < minimum:
        raise ApiError(400, f"Face is too small ({faces[0].w}px). Move closer or use a larger photo.")
    return faces[0]


def compute_registration_embedding(
    img: np.ndarray, face: Face, recognizer: FaceRecognizer
) -> np.ndarray | None:
    return recognizer.embed(img, face) if recognizer.available and face.raw is not None else None


def match_face(
    img: np.ndarray, faces: list[Face], recognizer: FaceRecognizer
) -> dict:
    """Match the largest face in ``faces`` against every registered user in one pass."""
    if not recognizer.available:
        raise ApiError(503, "Face recognition model is missing. Run: python scripts/download_models.py")
    if not faces:
        return {"status": "no_face_detected", "message": "No face detected in the image"}
    if faces[0].raw is None:
        raise ApiError(503, "Face detector model is missing. Run: python scripts/download_models.py")

    users, matrix = database.load_embeddings()
    if not users:
        return {"status": "no match", "message": "No registered users in database", "users_count": 0}

    query = recognizer.embed(img, faces[0])
    accepted, closest = recognizer.best_match(query, users, matrix)

    if accepted is None:
        return {
            "status": "no match",
            "message": "No matching face found in database",
            "users_count": len(users),
            "closest_score": round(closest.score, 4),
            "engine": recognizer.engine,
        }
    return {
        "status": "match",
        "name": accepted.user["name"],
        "matched_user_id": accepted.user["id"],
        "matched_image": accepted.user["image_path"],
        "score": round(accepted.score, 4),
        "similarity": round(accepted.confidence * 100, 1),
        "confidence": f"{round(accepted.confidence * 100, 1)}%",
        "engine": recognizer.engine,
    }


def backfill_embeddings(detector: FaceDetector, recognizer: FaceRecognizer, settings: Settings) -> int:
    """Compute embeddings for users registered before recognition existed. Returns rows updated."""
    if not recognizer.available or detector.yunet is None:
        return 0
    computed: dict[int, np.ndarray] = {}
    for row in database.users_without_embedding():
        try:
            path = resolve_static_path(settings, row["image_path"])
        except ApiError:
            continue
        img = cv2.imread(str(path)) if path.exists() else None
        faces = detector.detect(img) if img is not None else []
        if faces:
            computed[row["id"]] = recognizer.embed(img, faces[0])
    database.set_embeddings(computed)
    return len(computed)
