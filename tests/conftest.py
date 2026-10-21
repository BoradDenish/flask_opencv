from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import database
from app.config import get_settings
from app.dependencies import camera_dep, detector_dep, recognizer_dep
from app.main import create_app
from app.services.detection import Face
from app.services.recognition import FaceRecognizer

# Stub identities are encoded in image brightness: pixel value 30 -> person 1, 60 -> person 2, ...
IDENTITY_STEP = 30
EMBED_DIM = 8


def identity_image(person: int, size: int = 200) -> np.ndarray:
    return np.full((size, size, 3), person * IDENTITY_STEP, dtype=np.uint8)


def encode_jpeg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


class StubDetector:
    """Reports `face_count` identical faces covering most of the image."""

    engine = "Stub"
    loaded = True
    yunet = object()

    def __init__(self):
        self.face_count = 1
        self.face_width = 150

    def detect(self, img):
        w = min(self.face_width, img.shape[1])
        raw = np.zeros(15, dtype=np.float32)
        return [Face(5, 5, w, w, 0.99, raw) for _ in range(self.face_count)]


class StubRecognizer(FaceRecognizer):
    """One-hot embeddings derived from the image brightness, so identities are deterministic."""

    def __init__(self, settings):
        self.settings = settings
        self.model = object()

    def embed(self, img_bgr, face):
        person = int(round(float(img_bgr.mean()) / IDENTITY_STEP))
        vec = np.zeros(EMBED_DIM, dtype=np.float32)
        vec[person % EMBED_DIM] = 1.0
        return vec


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Point the database, static dir and models at throwaway locations."""
    monkeypatch.setenv("DB_FILE", str(tmp_path / "test_users.db"))
    monkeypatch.setenv("STATIC_DIR", str(tmp_path / "static"))
    database.init_db()


@pytest.fixture(autouse=True)
def no_deepface(monkeypatch):
    """Keep tests offline: DeepFace would download hundreds of MB of weights on first use."""
    monkeypatch.setattr("app.services.analysis.DEEPFACE_AVAILABLE", False)


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def detector():
    return StubDetector()


@pytest.fixture
def recognizer(settings):
    return StubRecognizer(settings)


@pytest.fixture
def camera(sample_frame):
    cam = MagicMock()
    cam.is_active.return_value = True
    cam.start.return_value = True
    cam.capture_frame.return_value = sample_frame.copy()
    cam.gen_frames.return_value = iter([b"--frame\r\nContent-Type: image/jpeg\r\n\r\nx\r\n"])
    return cam


@pytest.fixture
def client(detector, recognizer, camera):
    app = create_app()
    app.dependency_overrides[detector_dep] = lambda: detector
    app.dependency_overrides[recognizer_dep] = lambda: recognizer
    app.dependency_overrides[camera_dep] = lambda: camera
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_frame():
    return identity_image(1)
