"""Smoke tests against the real ONNX models; skipped until scripts/download_models.py has run."""
import numpy as np
import pytest

from app.config import get_settings
from app.services.detection import FaceDetector
from app.services.recognition import FaceRecognizer

pytestmark = pytest.mark.skipif(
    not (get_settings().yunet_path.exists() and get_settings().sface_path.exists()),
    reason="ONNX models not downloaded",
)


def test_yunet_finds_nothing_in_blank_image():
    detector = FaceDetector(get_settings())
    assert detector.engine == "YuNet"
    assert detector.detect(np.zeros((480, 640, 3), np.uint8)) == []


def test_sface_loads():
    assert FaceRecognizer(get_settings()).available
