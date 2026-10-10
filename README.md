# FastAPI OpenCV & Face Recognition System

A high-performance **FastAPI** web service for live webcam video streaming, Haar Cascade face & eye detection, photo capturing, database management, and facial verification/analysis (with DeepFace and OpenCV fallback support).

---

## 🚀 Features

- **FastAPI Core**: Async, non-blocking architecture powered by Starlette & Uvicorn.
- **Thread-Safe Camera Manager**: Handles webcam initialization, frame streaming, capture, and teardown cleanly with concurrency locks.
- **Live Video Streaming**: Real-time MJPEG video stream over `/video` with Haar Cascade face and eye detection overlays.
- **Face Registration & Photo Capture**: Captures webcam frames on demand, writes image files to `/static`, and stores records in SQLite (`users.db`).
- **Live Face Matching**: Verifies the current webcam frame against all registered database users.
- **Photo Analysis**: Analyzes facial attributes (age, gender, dominant emotion) using DeepFace (with graceful OpenCV Haar fallback if running under Python environments where TensorFlow/DeepFace is not installed).
- **Modern Responsive Dashboard**: Interactive web UI with camera controls, real-time status badges, instant capture preview, analysis badges, and registered face gallery management.

---

## 📦 Installation & Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. *(Optional)* If using **Python 3.9 - 3.12** and you want TensorFlow DeepFace AI models:
   ```bash
   pip install deepface tensorflow
   ```
   > *Note: On Python 3.14+, DeepFace is optional as TensorFlow prebuilt wheels are not yet released; the app automatically falls back to OpenCV Haar Cascade & histogram matching without crashing.*

---

## 🏃 Running the Application

You can launch the FastAPI server using any of the following commands:

```bash
# Recommended: Run directly with Uvicorn
uvicorn main:app --host 127.0.0.1 --port 5000 --reload

# Or via Python script
python main.py

# Or via backward-compatible app.py
python app.py
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

The project includes an automated test suite with **69 tests** covering database operations, camera lifecycle & frame streaming, face recognition & analysis algorithms, file upload face detection, and all FastAPI REST endpoints.

Run the test suite using `pytest`:

```bash
# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Run all tests with verbose output
pytest -v

# Run tests with code coverage report
pytest -v --cov=main --cov-report=term-missing

# Run specific test modules
pytest tests/test_api.py -v
pytest tests/test_camera.py -v
pytest tests/test_face_recognition.py -v
pytest tests/test_database.py -v
```

---

## 📁 Project Structure

```
flask_opencv/
├── Haarcascades/
│   ├── haarcascade_frontalface_default.xml
│   ├── haarcascade_eye.xml
│   ├── haarcascade_car.xml
│   └── haarcascade_fullbody.xml
├── static/              # Saved photos and temporary live frames
├── templates/
│   └── index.html       # Modern responsive dashboard template
├── tests/               # Comprehensive automated test suite (50 tests)
│   ├── __init__.py
│   ├── conftest.py      # Fixtures: isolated DB, mock camera, test client
│   ├── test_api.py      # FastAPI endpoints and streaming tests
│   ├── test_camera.py   # CameraManager concurrency and lifecycle tests
│   ├── test_database.py # SQLite CRUD and schema tests
│   └── test_face_recognition.py # Haar & DeepFace feature algorithms
├── app.py               # Compatibility launcher
├── main.py              # Main FastAPI application
├── pytest.ini           # Pytest test runner configuration
├── requirements.txt     # Python requirements (app & test dependencies)
├── users.db             # SQLite database for registered users
└── README.md            # Documentation
```
