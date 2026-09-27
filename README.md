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
├── app.py               # Compatibility launcher
├── main.py              # Main FastAPI application
├── requirements.txt     # Python requirements
├── users.db             # SQLite database for registered users
└── README.md            # Documentation
```
