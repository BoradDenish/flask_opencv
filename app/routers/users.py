import asyncio
import logging

from fastapi import APIRouter, Depends, File, Form, UploadFile
from starlette.concurrency import run_in_threadpool

from app import database
from app.config import Settings
from app.dependencies import camera_dep, detector_dep, recognizer_dep, settings_dep
from app.errors import ApiError
from app.services import faces, storage
from app.services.camera import CameraManager
from app.services.detection import FaceDetector
from app.services.recognition import FaceRecognizer

logger = logging.getLogger("face_service.users")
router = APIRouter(tags=["users"])


def register(
    name: str, img, detector: FaceDetector, recognizer: FaceRecognizer, settings: Settings
) -> dict:
    """Validate that the photo holds one usable face, then store the photo and its embedding."""
    face = faces.require_single_face(img, detector, settings)
    embedding = faces.compute_registration_embedding(img, face, recognizer)
    rel_path = storage.save_user_photo(settings, name, img)
    user_id = database.add_user(name, rel_path, embedding)
    return {
        "status": "photo captured",
        "file_path": rel_path,
        "user_id": user_id,
        "name": name,
        "faces_detected": 1,
        "recognition_ready": embedding is not None,
    }


@router.post("/capture_photo")
async def capture_photo(
    name: str = Form(...),
    camera: CameraManager = Depends(camera_dep),
    detector: FaceDetector = Depends(detector_dep),
    recognizer: FaceRecognizer = Depends(recognizer_dep),
    settings: Settings = Depends(settings_dep),
):
    """Capture the current webcam frame and register the user from it."""
    clean = storage.clean_name(name)
    if not camera.is_active():
        if not await run_in_threadpool(camera.start, settings.camera_index):
            raise ApiError(400, "Camera is not active or failed to start")
        await asyncio.sleep(0.3)  # let the sensor settle on auto-exposure

    frame = await run_in_threadpool(camera.capture_frame)
    if frame is None:
        raise ApiError(400, "Failed to capture photo from camera")
    return await run_in_threadpool(register, clean, frame, detector, recognizer, settings)


@router.post("/upload_register")
async def upload_register(
    name: str = Form(...),
    file: UploadFile = File(...),
    detector: FaceDetector = Depends(detector_dep),
    recognizer: FaceRecognizer = Depends(recognizer_dep),
    settings: Settings = Depends(settings_dep),
):
    """Register a user from an uploaded photo containing exactly one face."""
    clean = storage.clean_name(name)
    if not file.filename:
        raise ApiError(400, "Image file is required")
    img = storage.decode_image(await file.read())
    return await run_in_threadpool(register, clean, img, detector, recognizer, settings)


@router.get("/users")
async def list_users():
    return {"users": await run_in_threadpool(database.list_users)}


@router.delete("/users/{user_id}")
async def delete_user(user_id: int, settings: Settings = Depends(settings_dep)):
    image_path = await run_in_threadpool(database.delete_user, user_id)
    if image_path is None:
        raise ApiError(404, "User not found")
    try:
        storage.resolve_static_path(settings, image_path).unlink(missing_ok=True)
    except ApiError:
        logger.warning("Not removing photo outside static dir: %s", image_path)
    return {"status": "user deleted", "user_id": user_id}
