import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import numpy as np
import cv2
from starlette.testclient import TestClient

import main
from main import app, camera_manager


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """
    Provides an isolated SQLite database for testing to ensure
    production users.db is not modified.
    """
    test_db_path = tmp_path / "test_users.db"
    orig_db_file = main.DB_FILE
    monkeypatch.setattr(main, "DB_FILE", test_db_path)
    main.init_db()
    yield test_db_path
    monkeypatch.setattr(main, "DB_FILE", orig_db_file)


@pytest.fixture
def client(isolated_db):
    """
    TestClient with isolated DB and guaranteed camera cleanup.
    """
    camera_manager.stop()
    with TestClient(app) as test_client:
        yield test_client
    camera_manager.stop()


@pytest.fixture
def sample_frame():
    """
    Returns a sample BGR frame. Uses 'static/john wick.jpg' if present,
    or falls back to a synthetic 480x640x3 image.
    """
    sample_path = main.STATIC_DIR / "john wick.jpg"
    if sample_path.exists():
        img = cv2.imread(str(sample_path))
        if img is not None:
            return img.copy()
    
    # Synthetic frame fallback
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw simple shapes
    cv2.circle(frame, (320, 240), 80, (200, 200, 200), -1)
    return frame


@pytest.fixture
def mock_video_capture(sample_frame):
    """
    Creates a mock cv2.VideoCapture object.
    """
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, sample_frame.copy())
    mock_cap.release.return_value = None
    return mock_cap
