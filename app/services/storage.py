import re
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from app.config import Settings
from app.errors import ApiError

TMP_MAX_AGE_SECONDS = 600


def clean_name(raw: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_\- ]", "", raw).strip()
    if not name:
        raise ApiError(400, "Valid name is required")
    return name


def decode_image(contents: bytes) -> np.ndarray:
    if not contents:
        raise ApiError(400, "Uploaded file is empty")
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ApiError(400, "Uploaded file is not a valid image format")
    return img


def save_user_photo(settings: Settings, name: str, img: np.ndarray) -> str:
    """Write a registered photo under static/ and return its web-relative path."""
    settings.static_dir.mkdir(parents=True, exist_ok=True)
    # The suffix keeps two users with the same name from overwriting each other's photo.
    file_name = f"{name}_{uuid.uuid4().hex[:8]}.jpg"
    if not cv2.imwrite(str(settings.static_dir / file_name), img):
        raise ApiError(500, "Failed to save photo")
    return f"static/{file_name}"


def save_temp_image(settings: Settings, prefix: str, img: np.ndarray) -> str:
    """Write a short-lived preview image and return its web-relative path."""
    settings.tmp_dir.mkdir(parents=True, exist_ok=True)
    prune_temp_images(settings.tmp_dir)
    file_name = f"{prefix}_{uuid.uuid4().hex[:10]}.jpg"
    cv2.imwrite(str(settings.tmp_dir / file_name), img)
    return f"static/tmp/{file_name}"


def prune_temp_images(tmp_dir: Path) -> None:
    cutoff = time.time() - TMP_MAX_AGE_SECONDS
    for path in tmp_dir.glob("*.jpg"):
        if path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)


def resolve_static_path(settings: Settings, web_path: str) -> Path:
    """Resolve a user-supplied image path, refusing anything outside the static directory."""
    cleaned = web_path.strip().replace("\\", "/").lstrip("/")
    if cleaned.startswith("static/"):
        cleaned = cleaned[len("static/"):]
    root = settings.static_dir.resolve()
    target = (root / cleaned).resolve()
    if not target.is_relative_to(root):
        raise ApiError(400, "Image path must be inside the static directory")
    return target
