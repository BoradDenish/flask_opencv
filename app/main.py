import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import database
from app.config import get_settings
from app.errors import register_error_handlers
from app.routers import camera, faces, pages, users
from app.services.camera import CameraManager
from app.services.detection import FaceDetector
from app.services.faces import backfill_embeddings
from app.services.recognition import FaceRecognizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("face_service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.static_dir.mkdir(parents=True, exist_ok=True)
    database.init_db()

    detector = FaceDetector(settings)
    recognizer = FaceRecognizer(settings)
    app.state.detector = detector
    app.state.recognizer = recognizer
    app.state.camera = CameraManager(detector)

    updated = backfill_embeddings(detector, recognizer, settings)
    if updated:
        logger.info("Computed embeddings for %d existing users.", updated)
    yield
    app.state.camera.stop()


def create_app() -> FastAPI:
    settings = get_settings()
    settings.static_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title="OpenCV Face Recognition API",
        description="Webcam streaming, YuNet face detection and SFace face recognition.",
        version="3.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_error_handlers(app)

    app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
    app.state.templates = Jinja2Templates(directory=str(settings.templates_dir))

    for module in (pages, camera, users, faces):
        app.include_router(module.router)
    return app


app = create_app()
