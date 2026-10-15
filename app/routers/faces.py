import asyncio
import logging

import cv2
from fastapi import APIRouter, Depends, File, Form, UploadFile
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.dependencies import camera_dep, detector_dep, recognizer_dep, settings_dep
from app.errors import ApiError
from app.services import analysis, faces, storage
from app.services.camera import CameraManager
from app.services.detection import FaceDetector, annotate
from app.services.recognition import FaceRecognizer

logger = logging.getLogger("face_service.faces")
router = APIRouter(tags=["faces"])


def attributes_summary(raw: dict) -> dict:
    return {
        "age": raw.get("age"),
        "gender": raw.get("gender"),
        "emotion": (raw.get("emotion") or {}).get("dominant_emotion"),
        "engine": raw.get("engine"),
    }


def identify(
    img, detector: FaceDetector, recognizer: FaceRecognizer, settings: Settings, prefix: str
) -> dict:
    """Detect faces in an image, match the largest against the database and add attributes."""
    detected = detector.detect(img)
    result = faces.match_face(img, detected, recognizer)
    result["faces_detected"] = len(detected)
    result["annotated_path"] = storage.save_temp_image(settings, prefix, annotate(img, detected))
    if detected:
        try:
            raw = analysis.analyze_attributes(img, detector.engine, len(detected))
            result["analysis"] = attributes_summary(raw)
        except Exception as exc:
            logger.warning("Attribute analysis failed: %s", exc)
    return result


@router.post("/match_live_face")
async def match_live_face(
    camera: CameraManager = Depends(camera_dep),
    detector: FaceDetector = Depends(detector_dep),
    recognizer: FaceRecognizer = Depends(recognizer_dep),
    settings: Settings = Depends(settings_dep),
):
    """Capture a webcam frame and match the face in it against registered users."""
    if not camera.is_active():
        if not await run_in_threadpool(camera.start, settings.camera_index):
            raise ApiError(400, "Camera is not active or failed to capture live photo")
        await asyncio.sleep(0.3)

    frame = await run_in_threadpool(camera.capture_frame)
    if frame is None:
        raise ApiError(400, "Failed to capture live frame from camera")

    result = await run_in_threadpool(identify, frame, detector, recognizer, settings, "live")
    result["live_image"] = result.pop("annotated_path")
    return result


@router.post("/detect_upload")
async def detect_upload(
    file: UploadFile = File(...),
    detector: FaceDetector = Depends(detector_dep),
    recognizer: FaceRecognizer = Depends(recognizer_dep),
    settings: Settings = Depends(settings_dep),
):
    """Detect, analyse and match the faces in an uploaded image."""
    if not file.filename:
        raise ApiError(400, "Image file is required")
    img = storage.decode_image(await file.read())

    uploaded_path = await run_in_threadpool(storage.save_temp_image, settings, "upload", img)
    result = await run_in_threadpool(identify, img, detector, recognizer, settings, "annotated")
    result["annotated_image"] = result.pop("annotated_path")
    result["uploaded_image"] = uploaded_path
    return result


@router.post("/analyze_photo")
async def analyze_photo(
    image_path: str = Form(...),
    detector: FaceDetector = Depends(detector_dep),
    settings: Settings = Depends(settings_dep),
):
    """Analyse a stored photo (path relative to the static directory)."""
    if not image_path.strip():
        raise ApiError(400, "Image path is required")
    target = storage.resolve_static_path(settings, image_path)
    if not target.is_file():
        raise ApiError(404, f"Image file not found: {image_path.strip()}")

    img = await run_in_threadpool(cv2.imread, str(target))
    if img is None:
        raise ApiError(400, "File is not a readable image")

    def run():
        return analysis.analyze_attributes(img, detector.engine, len(detector.detect(img)))

    try:
        return {"analysis": await run_in_threadpool(run)}
    except Exception as exc:
        logger.error("analyze_photo failed: %s", exc)
        raise ApiError(500, str(exc))
