from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

from app import database
from app.dependencies import camera_dep, detector_dep, recognizer_dep
from app.services.analysis import DEEPFACE_AVAILABLE
from app.services.camera import CameraManager
from app.services.detection import FaceDetector
from app.services.recognition import FaceRecognizer

router = APIRouter()


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request, camera: CameraManager = Depends(camera_dep)):
    users = await run_in_threadpool(database.list_users)
    return request.app.state.templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "users": users,
            "deepface_available": DEEPFACE_AVAILABLE,
            "camera_active": camera.is_active(),
        },
    )


@router.get("/api/status", tags=["system"])
async def system_status(
    camera: CameraManager = Depends(camera_dep),
    detector: FaceDetector = Depends(detector_dep),
    recognizer: FaceRecognizer = Depends(recognizer_dep),
):
    count = await run_in_threadpool(database.count_users)
    return {
        "status": "online",
        "camera_active": camera.is_active(),
        "deepface_available": DEEPFACE_AVAILABLE,
        "detector": detector.engine,
        "recognizer_loaded": recognizer.available,
        "face_cascade_loaded": detector.loaded,
        "registered_users_count": count,
    }
