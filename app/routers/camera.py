from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.dependencies import camera_dep, settings_dep
from app.errors import ApiError
from app.services.camera import CameraManager

router = APIRouter(tags=["camera"])


@router.get("/video")
async def video(camera: CameraManager = Depends(camera_dep)):
    """Multipart MJPEG stream with face detection overlays."""
    if not camera.is_active():
        raise ApiError(400, "Camera is off")
    return StreamingResponse(
        camera.gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/start_camera")
async def start_camera(
    camera: CameraManager = Depends(camera_dep), settings: Settings = Depends(settings_dep)
):
    if not await run_in_threadpool(camera.start, settings.camera_index):
        raise ApiError(500, "Unable to start webcam")
    return JSONResponse({"status": "camera started"})


@router.get("/stop_camera")
async def stop_camera(camera: CameraManager = Depends(camera_dep)):
    await run_in_threadpool(camera.stop)
    return JSONResponse({"status": "camera stopped"})
