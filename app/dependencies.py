from fastapi import Request

from app.config import Settings, get_settings
from app.services.camera import CameraManager
from app.services.detection import FaceDetector
from app.services.recognition import FaceRecognizer


def settings_dep() -> Settings:
    return get_settings()


def detector_dep(request: Request) -> FaceDetector:
    return request.app.state.detector


def recognizer_dep(request: Request) -> FaceRecognizer:
    return request.app.state.recognizer


def camera_dep(request: Request) -> CameraManager:
    return request.app.state.camera
