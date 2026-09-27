import os
import re
import sys
import time
import sqlite3
import threading
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, Request, Form, HTTPException, status
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("fastapi_opencv")

# Base paths
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
HAARCASCADES_DIR = BASE_DIR / "Haarcascades"
DB_FILE = BASE_DIR / "users.db"

# Ensure directories exist
STATIC_DIR.mkdir(parents=True, exist_ok=True)
HAARCASCADES_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

# DeepFace Import handling (graceful fallback if TensorFlow / DeepFace is unavailable)
DEEPFACE_AVAILABLE = False
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    logger.info("DeepFace loaded successfully.")
except Exception as exc:
    DEEPFACE_AVAILABLE = False
    logger.warning(f"DeepFace not available ({exc}). Falling back to OpenCV-based detection & matching.")

# Load Haar Cascades
face_cascade_path = HAARCASCADES_DIR / "haarcascade_frontalface_default.xml"
eye_cascade_path = HAARCASCADES_DIR / "haarcascade_eye.xml"

face_cascade: Optional[cv2.CascadeClassifier] = None
eye_cascade: Optional[cv2.CascadeClassifier] = None

if hasattr(cv2, "CascadeClassifier"):
    if face_cascade_path.exists():
        fc = cv2.CascadeClassifier(str(face_cascade_path))
        if not fc.empty():
            face_cascade = fc
            logger.info("Loaded face cascade successfully.")
    if eye_cascade_path.exists():
        ec = cv2.CascadeClassifier(str(eye_cascade_path))
        if not ec.empty():
            eye_cascade = ec
            logger.info("Loaded eye cascade successfully.")

# Database Helpers
def get_db():
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                image_path TEXT NOT NULL
            )
        """)
        conn.commit()
    logger.info(f"Database initialized at {DB_FILE}")

# Thread-Safe Camera Manager
class CameraManager:
    def __init__(self):
        self.camera: Optional[cv2.VideoCapture] = None
        self.lock = threading.Lock()
        self.is_running = False

    def start(self, camera_index: int = 0) -> bool:
        with self.lock:
            if self.camera is not None and self.camera.isOpened():
                self.is_running = True
                return True
            
            # Use CAP_DSHOW on Windows for fast webcam initialization
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            cap = cv2.VideoCapture(camera_index, backend)
            if not cap.isOpened():
                # Fallback to default backend
                cap = cv2.VideoCapture(camera_index)

            if cap.isOpened():
                self.camera = cap
                self.is_running = True
                logger.info(f"Camera opened successfully (index={camera_index}).")
                return True
            else:
                self.camera = None
                self.is_running = False
                logger.error(f"Failed to open camera (index={camera_index}).")
                return False

    def stop(self) -> bool:
        with self.lock:
            self.is_running = False
            if self.camera is not None:
                try:
                    self.camera.release()
                except Exception as e:
                    logger.error(f"Error releasing camera: {e}")
                finally:
                    self.camera = None
            logger.info("Camera stopped.")
            return True

    def is_active(self) -> bool:
        with self.lock:
            return self.camera is not None and self.camera.isOpened() and self.is_running

    def capture_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.camera is not None and self.camera.isOpened():
                success, frame = self.camera.read()
                if success and frame is not None:
                    return frame.copy()
            return None

    def gen_frames(self):
        while True:
            # Check state inside lock
            with self.lock:
                if not self.is_running or self.camera is None or not self.camera.isOpened():
                    break
                success, frame = self.camera.read()

            if not success or frame is None:
                time.sleep(0.05)
                continue

            # Haar cascade face & eye detection
            try:
                if face_cascade is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 7)

                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        
                        if eye_cascade is not None:
                            roi_gray = gray[y:y + h, x:x + w]
                            roi_color = frame[y:y + h, x:x + w]
                            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                            for (ex, ey, ew, eh) in eyes:
                                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            except Exception as e:
                logger.warning(f"Detection error during streaming: {e}")

            # Encode frame to JPEG
            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            # Regulate frame rate to ~30 FPS and yield control
            time.sleep(0.03)

camera_manager = CameraManager()

# Face Analysis & Verification Functions
def extract_face_hist(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Fallback histogram feature extractor for face matching without DeepFace."""
    if face_cascade is None:
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face_roi = cv2.resize(gray[y:y+h, x:x+w], (128, 128))
    hist = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def perform_face_analysis(img_path: str) -> Dict[str, Any]:
    """Analyze face for age, gender, emotion using DeepFace (with OpenCV fallback)."""
    if DEEPFACE_AVAILABLE:
        analysis = DeepFace.analyze(img_path=img_path, actions=['age', 'gender', 'emotion'])
        item = analysis[0] if isinstance(analysis, list) else analysis
        
        # Format gender string cleanly
        gender = item.get("dominant_gender") or item.get("gender")
        if isinstance(gender, dict):
            gender = max(gender, key=gender.get)

        emotion = item.get("dominant_emotion")
        if not emotion and isinstance(item.get("emotion"), dict):
            emotion = max(item["emotion"], key=item["emotion"].get)

        return {
            "age": item.get("age"),
            "gender": gender,
            "emotion": {
                "dominant_emotion": emotion,
                "details": item.get("emotion")
            },
            "engine": "DeepFace",
            "raw": item
        }
    else:
        # Fallback using OpenCV Haar Cascades
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image at {img_path}")
        
        faces_count = 0
        eyes_count = 0
        if face_cascade is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            faces_count = len(faces)
            if eye_cascade is not None and faces_count > 0:
                for (x, y, w, h) in faces:
                    eyes = eye_cascade.detectMultiScale(gray[y:y+h, x:x+w], 1.1, 3)
                    eyes_count += len(eyes)

        return {
            "age": "N/A (DeepFace not loaded)",
            "gender": "N/A (DeepFace not loaded)",
            "emotion": {
                "dominant_emotion": "Neutral / Detected",
                "details": {"faces_detected": faces_count, "eyes_detected": eyes_count}
            },
            "faces_detected": faces_count,
            "engine": "OpenCV Haar Cascades (Fallback)",
            "note": "DeepFace requires TensorFlow (Python <= 3.12). Currently running OpenCV fallback."
        }

def perform_face_verification(live_path: str, candidate_path: str) -> Dict[str, Any]:
    """Verify live face against registered face using DeepFace (with OpenCV fallback)."""
    if DEEPFACE_AVAILABLE:
        result = DeepFace.verify(img1_path=live_path, img2_path=candidate_path)
        return {
            "verified": bool(result.get("verified", False)),
            "distance": float(result.get("distance", 0.0)),
            "threshold": float(result.get("threshold", 0.0)),
            "engine": "DeepFace"
        }
    else:
        # Fallback face comparison via histogram correlation
        live_img = cv2.imread(live_path)
        cand_img = cv2.imread(candidate_path)
        if live_img is None or cand_img is None:
            return {"verified": False, "distance": 1.0, "engine": "OpenCV (Read Error)"}

        h1 = extract_face_hist(live_img)
        h2 = extract_face_hist(cand_img)
        if h1 is None or h2 is None:
            return {"verified": False, "distance": 1.0, "engine": "OpenCV (No face detected)"}

        similarity = float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
        # Correlation >= 0.70 is considered a match in normalized grayscale histogram
        return {
            "verified": similarity >= 0.70,
            "distance": round(1.0 - similarity, 4),
            "similarity": round(similarity, 4),
            "engine": "OpenCV Histogram Fallback"
        }

# Lifespan context manager for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield
    camera_manager.stop()

# FastAPI App Instance
app = FastAPI(
    title="OpenCV & DeepFace Facial Recognition API",
    description="FastAPI service for webcam streaming, face/eye detection, photo capture, and face verification.",
    version="2.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files & Jinja2 Templates
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Routes

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main interactive frontend dashboard."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, image_path FROM users ORDER BY id DESC")
        users = cursor.fetchall()
        user_list = [dict(u) for u in users]
    
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "users": user_list,
            "deepface_available": DEEPFACE_AVAILABLE,
            "camera_active": camera_manager.is_active()
        }
    )

@app.get("/video")
async def video():
    """Video streaming route yielding multipart/x-mixed-replace JPEG frames."""
    if camera_manager.is_active():
        return StreamingResponse(
            camera_manager.gen_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Camera is off"}
        )

@app.get("/start_camera")
async def start_camera():
    """Start the webcam video capture."""
    success = await run_in_threadpool(camera_manager.start)
    if success:
        return JSONResponse({"status": "camera started"})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Unable to start webcam"}
    )

@app.get("/stop_camera")
async def stop_camera():
    """Stop the webcam video capture."""
    await run_in_threadpool(camera_manager.stop)
    return JSONResponse({"status": "camera stopped"})

@app.post("/capture_photo")
async def capture_photo(name: str = Form(...)):
    """Capture the current webcam frame, save it, and register the user in the database."""
    clean_name = re.sub(r'[^a-zA-Z0-9_\- ]', '', name).strip()
    if not clean_name:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Valid name is required"}
        )

    if not camera_manager.is_active():
        # Try auto-starting if not active
        started = await run_in_threadpool(camera_manager.start)
        if not started:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Camera is not active or failed to start"}
            )
        # Give camera sensor a brief moment to settle
        time.sleep(0.3)

    frame = await run_in_threadpool(camera_manager.capture_frame)
    if frame is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Failed to capture photo from camera"}
        )

    file_name = f"{clean_name}.jpg"
    abs_file_path = STATIC_DIR / file_name
    rel_file_path = f"static/{file_name}"

    # Save to disk
    await run_in_threadpool(cv2.imwrite, str(abs_file_path), frame)

    # Insert into database
    def _save_to_db():
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO users (name, image_path) VALUES (?, ?)", (clean_name, rel_file_path))
            user_id = cur.lastrowid
            conn.commit()
            return user_id

    user_id = await run_in_threadpool(_save_to_db)

    return JSONResponse(
        content={
            "status": "photo captured",
            "file_path": rel_file_path,
            "user_id": user_id,
            "name": clean_name
        }
    )

@app.post("/analyze_photo")
async def analyze_photo(image_path: str = Form(...)):
    """Analyze a photo for age, gender, and facial emotion."""
    clean_path = image_path.strip().replace("\\", "/")
    if not clean_path:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Image path is required"}
        )

    target_path = BASE_DIR / clean_path if not Path(clean_path).is_absolute() else Path(clean_path)

    if not target_path.exists():
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": f"Image file not found: {clean_path}"}
        )

    try:
        analysis_result = await run_in_threadpool(perform_face_analysis, str(target_path))
        return JSONResponse({"analysis": analysis_result})
    except Exception as e:
        logger.error(f"Error in analyze_photo: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )

@app.post("/match_live_face")
async def match_live_face():
    """Capture live frame and verify against stored user images in database."""
    if not camera_manager.is_active():
        started = await run_in_threadpool(camera_manager.start)
        if not started:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Camera is not active or failed to capture live photo"}
            )
        time.sleep(0.3)

    frame = await run_in_threadpool(camera_manager.capture_frame)
    if frame is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Failed to capture live frame from camera"}
        )

    live_file_path = STATIC_DIR / "live_temp.jpg"
    await run_in_threadpool(cv2.imwrite, str(live_file_path), frame)

    # Fetch users from database
    def _fetch_users():
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, image_path FROM users")
            return cursor.fetchall()

    users = await run_in_threadpool(_fetch_users)
    if not users:
        return JSONResponse(
            content={"status": "no match", "message": "No registered users in database"}
        )

    def _verify_all():
        for user in users:
            name, img_path = user["name"], user["image_path"]
            abs_cand_path = BASE_DIR / img_path.replace("\\", "/")
            if not abs_cand_path.exists():
                continue

            try:
                result = perform_face_verification(str(live_file_path), str(abs_cand_path))
                if result.get("verified"):
                    return {
                        "status": "match",
                        "name": name,
                        "details": result
                    }
            except Exception as e:
                logger.warning(f"Verification error against user {name}: {e}")
                continue

        return {"status": "no match"}

    result = await run_in_threadpool(_verify_all)
    return JSONResponse(result)

@app.get("/users")
async def list_users():
    """Retrieve all registered users and photo paths."""
    def _get_users():
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, name, image_path FROM users ORDER BY id DESC")
            return [dict(row) for row in cur.fetchall()]

    user_list = await run_in_threadpool(_get_users)
    return JSONResponse({"users": user_list})

@app.delete("/users/{user_id}")
async def delete_user(user_id: int):
    """Delete a registered user and optionally remove their photo."""
    def _delete():
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT image_path FROM users WHERE id = ?", (user_id,))
            row = cur.fetchone()
            if not row:
                return False
            cur.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            return True

    success = await run_in_threadpool(_delete)
    if success:
        return JSONResponse({"status": "user deleted", "user_id": user_id})
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": "User not found"}
    )

@app.get("/api/status")
async def system_status():
    """System health and diagnostic status."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) as count FROM users")
        count = cur.fetchone()["count"]

    return JSONResponse({
        "status": "online",
        "camera_active": camera_manager.is_active(),
        "deepface_available": DEEPFACE_AVAILABLE,
        "face_cascade_loaded": face_cascade is not None,
        "eye_cascade_loaded": eye_cascade is not None,
        "registered_users_count": count
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)
