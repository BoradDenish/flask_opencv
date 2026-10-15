# FastAPI OpenCV & Face Recognition System

A high-performance **FastAPI** web service for live webcam video streaming, Haar Cascade face & eye detection, photo capturing, database management, and facial verification/analysis (with DeepFace and OpenCV fallback support).

---

## 🚀 Features

- **FastAPI Core**: Async, non-blocking architecture powered by Starlette & Uvicorn.
- **Thread-Safe Camera Manager**: Handles webcam initialization, frame streaming, capture, and teardown cleanly with concurrency locks.
- **Live Video Streaming**: Real-time MJPEG video stream over `/video` with Haar Cascade face and eye detection overlays.
- **Face Registration & Photo Capture**: Captures webcam frames on demand, writes image files to `/static`, and stores records in SQLite (`users.db`).
- **Accurate Face Recognition**: YuNet detects faces and landmarks, SFace turns each aligned face into a 128-d embedding, and matching is cosine similarity against stored embeddings (one vectorised pass, no per-user loop). No TensorFlow needed.
- **Photo Analysis**: Age, gender and emotion come from DeepFace when installed; otherwise they are reported as unavailable rather than guessed.
- **Modern Responsive Dashboard**: Interactive web UI with camera controls, real-time status badges, instant capture preview, analysis badges, and registered face gallery management.

---

## 📦 Installation & Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the face models** (YuNet detector + SFace recognizer, from the OpenCV Zoo):
   ```bash
   python scripts/download_models.py
   ```
   Without them the app still runs (Haar detection) but `/match_live_face`, `/detect_upload` and registration embeddings return a clear error.

3. *(Optional)* Age / gender / emotion analysis (Python <= 3.12):
   ```bash
   pip install deepface tensorflow
   ```

### Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `DB_FILE` | `users.db` | SQLite database path |
| `STATIC_DIR` | `static/` | Where registered photos are stored |
| `MODELS_DIR` | `models/` | ONNX model location |
| `CAMERA_INDEX` | `0` | Webcam device index |
| `MATCH_THRESHOLD` | `0.363` | Cosine similarity needed to accept a match (OpenCV Zoo value for SFace; raise it for fewer false matches) |
| `DETECT_SCORE` | `0.9` | Minimum YuNet detection confidence |
| `MIN_FACE_SIZE` | `60` | Smallest face width (px) accepted at registration |

Registration requires exactly one clear face; users registered before this version get embeddings computed automatically at startup.

---

## 🏃 Running the Application

You can launch the FastAPI server using any of the following commands:

```bash
# Recommended: Run directly with Uvicorn
uvicorn app.main:app --host 127.0.0.1 --port 5000 --reload

# Or via Python script
python main.py
```

Once started, open your browser and navigate to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

Interactive API documentation is also available at:
👉 **[http://127.0.0.1:5000/docs](http://127.0.0.1:5000/docs)** (Swagger UI)

---

## 🔌 API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web dashboard UI |
| `GET` | `/video` | Multipart MJPEG webcam stream |
| `GET` | `/start_camera` | Starts webcam capture |
| `GET` | `/stop_camera` | Stops and releases webcam |
| `POST` | `/capture_photo` | Captures current frame and saves user (`name` form param) |
| `POST` | `/analyze_photo` | Analyzes image (`image_path` form param) for age, gender, emotion |
| `POST` | `/match_live_face` | Compares live webcam against registered users |
| `GET` | `/users` | Lists all registered users from SQLite database |
| `DELETE` | `/users/{user_id}` | Deletes a registered user from the database |
| `GET` | `/api/status` | Returns system diagnostic status and engine information |
| `POST` | `/detect_upload` | Detects faces, annotates bounding boxes, analyzes attributes, and matches uploaded file against DB |
| `POST` | `/upload_register` | Registers a new user directly from an uploaded photo (`name` form param, `file` upload) |

---

## 🧪 Running Tests

```bash
pytest -v
pytest -v --cov=app --cov-report=term-missing
```

Tests use stub detector/recognizer objects, so they run without a webcam; `tests/test_models.py` additionally smoke-tests the real ONNX models when they are downloaded.

---

## 📁 Project Structure

```
flask_opencv/
├── app/
│   ├── main.py            # create_app() + lifespan (loads models, camera)
│   ├── config.py          # Settings read from environment
│   ├── database.py        # SQLite access (users + embeddings)
│   ├── dependencies.py    # FastAPI dependency providers
│   ├── errors.py          # ApiError -> {"error": ...}
│   ├── routers/           # pages, camera, users, faces
│   └── services/          # detection, recognition, analysis, camera, storage, faces
├── Haarcascades/          # fallback detector
├── models/                # YuNet + SFace ONNX (downloaded, git-ignored)
├── scripts/download_models.py
├── static/                # registered photos, tmp/ previews (git-ignored)
├── templates/index.html
├── tests/
├── main.py                # entry point
└── requirements.txt
```
