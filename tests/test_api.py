import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import cv2
import numpy as np
import pytest

import main
from main import camera_manager


def test_index_dashboard(client):
    """Verify GET / renders dashboard HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Face" in response.text
    assert "Live Video Stream" in response.text or "Webcam" in response.text


def test_api_status(client):
    """Verify GET /api/status returns diagnostic details."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "camera_active" in data
    assert "deepface_available" in data
    assert "face_cascade_loaded" in data
    assert "registered_users_count" in data


def test_video_stream_inactive_camera(client):
    """Verify GET /video returns 400 when camera is not running."""
    camera_manager.stop()
    response = client.get("/video")
    assert response.status_code == 400
    assert response.json() == {"error": "Camera is off"}


def test_video_stream_active_camera(client):
    """Verify GET /video streams multipart frames when camera is active."""
    def fake_frames():
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nfake_frame_bytes\r\n"

    with patch.object(camera_manager, "is_active", return_value=True), \
         patch.object(camera_manager, "gen_frames", side_effect=fake_frames):
        response = client.get("/video")
        assert response.status_code == 200
        assert "multipart/x-mixed-replace" in response.headers["content-type"]
        assert b"--frame" in response.content


def test_start_camera_success(client):
    """Verify GET /start_camera returns 200 when webcam opens."""
    with patch.object(camera_manager, "start", return_value=True):
        response = client.get("/start_camera")
        assert response.status_code == 200
        assert response.json() == {"status": "camera started"}


def test_start_camera_failure(client):
    """Verify GET /start_camera returns 500 when webcam fails to open."""
    with patch.object(camera_manager, "start", return_value=False):
        response = client.get("/start_camera")
        assert response.status_code == 500
        assert "error" in response.json()


def test_stop_camera(client):
    """Verify GET /stop_camera stops the webcam and returns 200."""
    response = client.get("/stop_camera")
    assert response.status_code == 200
    assert response.json() == {"status": "camera stopped"}
    assert camera_manager.is_active() is False


def test_capture_photo_invalid_name(client):
    """Verify POST /capture_photo rejects empty, whitespace, or invalid names."""
    # Missing form field triggers 422
    res_missing = client.post("/capture_photo")
    assert res_missing.status_code == 422

    # Whitespace-only name triggers 400
    res_spaces = client.post("/capture_photo", data={"name": "   "})
    assert res_spaces.status_code == 400
    assert "error" in res_spaces.json()

    # Special characters only triggers 400
    res_special = client.post("/capture_photo", data={"name": "@#$%^"})
    assert res_special.status_code == 400
    assert "error" in res_special.json()


def test_capture_photo_camera_failure(client):
    """Verify POST /capture_photo returns 400 when camera fails to start or capture."""
    with patch.object(camera_manager, "is_active", return_value=False), \
         patch.object(camera_manager, "start", return_value=False):
        res = client.post("/capture_photo", data={"name": "Alice"})
        assert res.status_code == 400
        assert "error" in res.json()


def test_capture_photo_success(client, sample_frame):
    """Verify POST /capture_photo saves image file and registers user in DB."""
    test_user_name = "TestRunnerUnit"
    created_file = main.STATIC_DIR / f"{test_user_name}.jpg"

    try:
        with patch.object(camera_manager, "is_active", return_value=True), \
             patch.object(camera_manager, "capture_frame", return_value=sample_frame.copy()):
            res = client.post("/capture_photo", data={"name": test_user_name})
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "photo captured"
            assert data["name"] == test_user_name
            assert "user_id" in data
            assert created_file.exists()

            # Verify in DB
            with main.get_db() as conn:
                row = conn.execute("SELECT * FROM users WHERE id = ?", (data["user_id"],)).fetchone()
                assert row is not None
                assert row["name"] == test_user_name
    finally:
        if created_file.exists():
            created_file.unlink()


def test_analyze_photo_validation(client):
    """Verify POST /analyze_photo validates inputs."""
    # Missing form field triggers 422
    res_missing_field = client.post("/analyze_photo")
    assert res_missing_field.status_code == 422

    # Whitespace-only path triggers 400
    res_spaces = client.post("/analyze_photo", data={"image_path": "   "})
    assert res_spaces.status_code == 400

    # Non-existent file triggers 404
    res_missing_file = client.post("/analyze_photo", data={"image_path": "non_existent_file_abc.jpg"})
    assert res_missing_file.status_code == 404


def test_analyze_photo_success(client):
    """Verify POST /analyze_photo analyzes sample face image."""
    sample_path = "static/john wick.jpg"
    res = client.post("/analyze_photo", data={"image_path": sample_path})
    assert res.status_code == 200
    data = res.json()
    assert "analysis" in data
    assert "age" in data["analysis"]
    assert "gender" in data["analysis"]
    assert "emotion" in data["analysis"]


def test_analyze_photo_leading_slash_path(client):
    """Verify POST /analyze_photo handles leading slashes in path properly."""
    sample_path = "/static/john wick.jpg"
    res = client.post("/analyze_photo", data={"image_path": sample_path})
    assert res.status_code == 200
    data = res.json()
    assert "analysis" in data


def test_match_live_face_no_camera(client):
    """Verify POST /match_live_face handles camera startup failure."""
    with patch.object(camera_manager, "is_active", return_value=False), \
         patch.object(camera_manager, "start", return_value=False):
        res = client.post("/match_live_face")
        assert res.status_code == 400
        assert "error" in res.json()


def test_match_live_face_no_registered_users(client, sample_frame):
    """Verify POST /match_live_face reports no match when DB is empty."""
    with patch.object(camera_manager, "is_active", return_value=True), \
         patch.object(camera_manager, "capture_frame", return_value=sample_frame.copy()):
        res = client.post("/match_live_face")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "no match"
        assert "No registered users" in data.get("message", "")


def test_match_live_face_match_found(client, sample_frame):
    """Verify POST /match_live_face returns matching user info."""
    # Register user with same image
    with main.get_db() as conn:
        conn.execute("INSERT INTO users (name, image_path) VALUES (?, ?)", ("John Wick", "static/john wick.jpg"))
        conn.commit()

    with patch.object(camera_manager, "is_active", return_value=True), \
         patch.object(camera_manager, "capture_frame", return_value=sample_frame.copy()):
        res = client.post("/match_live_face")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "match"
        assert data["name"] == "John Wick"


def test_match_live_face_no_match(client, tmp_path):
    """Verify POST /match_live_face returns no match when face does not match."""
    # Create a blank image for candidate user
    blank_img = np.zeros((300, 300, 3), dtype=np.uint8)
    cand_file = main.STATIC_DIR / "blank_test_user.jpg"
    cv2.imwrite(str(cand_file), blank_img)

    try:
        with main.get_db() as conn:
            conn.execute("INSERT INTO users (name, image_path) VALUES (?, ?)", ("Blank Person", "static/blank_test_user.jpg"))
            conn.commit()

        # Capture frame with different face
        face_img = cv2.imread(str(main.STATIC_DIR / "john wick.jpg"))
        with patch.object(camera_manager, "is_active", return_value=True), \
             patch.object(camera_manager, "capture_frame", return_value=face_img.copy()):
            res = client.post("/match_live_face")
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "no match"
    finally:
        if cand_file.exists():
            cand_file.unlink()


def test_list_and_delete_users(client):
    """Verify GET /users and DELETE /users/{id} endpoints."""
    # List empty
    res = client.get("/users")
    assert res.status_code == 200
    assert res.json()["users"] == []

    # Insert user
    with main.get_db() as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO users (name, image_path) VALUES (?, ?)", ("Test Subject", "static/ts.jpg"))
        uid = cur.lastrowid
        conn.commit()

    # List has user
    res = client.get("/users")
    assert res.status_code == 200
    users = res.json()["users"]
    assert len(users) == 1
    assert users[0]["id"] == uid
    assert users[0]["name"] == "Test Subject"

    # Delete user
    res_del = client.delete(f"/users/{uid}")
    assert res_del.status_code == 200
    assert res_del.json()["status"] == "user deleted"

    # Delete non-existent user
    res_del_404 = client.delete(f"/users/{uid}")
    assert res_del_404.status_code == 404
    assert res_del_404.json()["error"] == "User not found"


def test_app_launcher_import():
    """Verify app.py imports app instance correctly."""
    import app as app_module
    assert hasattr(app_module, "app")
    assert app_module.app is main.app


def test_capture_photo_frame_none(client):
    """Verify POST /capture_photo handles frame capture failure."""
    with patch.object(camera_manager, "is_active", return_value=True), \
         patch.object(camera_manager, "capture_frame", return_value=None):
        res = client.post("/capture_photo", data={"name": "Alice"})
        assert res.status_code == 400
        assert "Failed to capture photo" in res.json()["error"]


def test_analyze_photo_internal_error(client):
    """Verify POST /analyze_photo handles internal analysis errors with 500."""
    with patch("main.perform_face_analysis", side_effect=RuntimeError("Model failure")):
        res = client.post("/analyze_photo", data={"image_path": "static/john wick.jpg"})
        assert res.status_code == 500
        assert "Model failure" in res.json()["error"]


def test_match_live_face_frame_none(client):
    """Verify POST /match_live_face handles failure when live frame cannot be captured."""
    with patch.object(camera_manager, "is_active", return_value=True), \
         patch.object(camera_manager, "capture_frame", return_value=None):
        res = client.post("/match_live_face")
        assert res.status_code == 400
        assert "Failed to capture live frame" in res.json()["error"]


def test_match_live_face_missing_candidate_file(client, sample_frame):
    """Verify POST /match_live_face skips missing candidate files cleanly."""
    with main.get_db() as conn:
        conn.execute("INSERT INTO users (name, image_path) VALUES (?, ?)", ("Ghost", "static/ghost_not_found.jpg"))
        conn.commit()

    with patch.object(camera_manager, "is_active", return_value=True), \
         patch.object(camera_manager, "capture_frame", return_value=sample_frame.copy()):
        res = client.post("/match_live_face")
        assert res.status_code == 200
        assert res.json()["status"] == "no match"


def test_match_live_face_verification_exception(client, sample_frame):
    """Verify POST /match_live_face handles verification exceptions cleanly."""
    with main.get_db() as conn:
        conn.execute("INSERT INTO users (name, image_path) VALUES (?, ?)", ("ErrorUser", "static/john wick.jpg"))
        conn.commit()

    with patch.object(camera_manager, "is_active", return_value=True), \
         patch.object(camera_manager, "capture_frame", return_value=sample_frame.copy()), \
         patch("main.perform_face_verification", side_effect=Exception("Corrupt image data")):
        res = client.post("/match_live_face")
        assert res.status_code == 200
        assert res.json()["status"] == "no match"


def test_detect_upload_no_file(client):
    """Verify POST /detect_upload returns 422 if file field is missing."""
    res = client.post("/detect_upload")
    assert res.status_code == 422


def test_detect_upload_empty_file(client):
    """Verify POST /detect_upload returns 400 for empty file contents."""
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    res = client.post("/detect_upload", files=files)
    assert res.status_code == 400
    assert "empty" in res.json()["error"]


def test_detect_upload_invalid_image(client):
    """Verify POST /detect_upload returns 400 for corrupted non-image content."""
    files = {"file": ("bad.txt", b"not-an-image", "text/plain")}
    res = client.post("/detect_upload", files=files)
    assert res.status_code == 400
    assert "not a valid image" in res.json()["error"]


def test_detect_upload_no_face(client):
    """Verify POST /detect_upload processes image without faces."""
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", blank)
    files = {"file": ("blank.jpg", encoded.tobytes(), "image/jpeg")}
    res = client.post("/detect_upload", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["faces_detected"] == 0
    assert data["status"] == "no match"


def test_detect_upload_with_matching_face(client, sample_frame):
    """Verify POST /detect_upload detects faces and matches registered database user."""
    with main.get_db() as conn:
        conn.execute("INSERT INTO users (name, image_path) VALUES (?, ?)", ("Keanu", "static/john wick.jpg"))
        conn.commit()

    _, encoded = cv2.imencode(".jpg", sample_frame)
    files = {"file": ("face.jpg", encoded.tobytes(), "image/jpeg")}
    res = client.post("/detect_upload", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "match"
    assert data["name"] == "Keanu"
    assert "annotated_image" in data
    assert "confidence" in data


def test_upload_register_validation(client):
    """Verify POST /upload_register input validation."""
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", blank)
    files = {"file": ("face.jpg", encoded.tobytes(), "image/jpeg")}
    res_no_name = client.post("/upload_register", files=files)
    assert res_no_name.status_code == 422

    res_spaces = client.post("/upload_register", data={"name": "   "}, files=files)
    assert res_spaces.status_code == 400

    res_no_file = client.post("/upload_register", data={"name": "Alice"})
    assert res_no_file.status_code == 422

    empty_files = {"file": ("empty.jpg", b"", "image/jpeg")}
    res_empty = client.post("/upload_register", data={"name": "Alice"}, files=empty_files)
    assert res_empty.status_code == 400

    bad_files = {"file": ("bad.txt", b"plain text", "text/plain")}
    res_bad = client.post("/upload_register", data={"name": "Alice"}, files=bad_files)
    assert res_bad.status_code == 400


def test_upload_register_no_face(client):
    """Verify POST /upload_register rejects photos without detected faces."""
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", blank)
    files = {"file": ("blank.jpg", encoded.tobytes(), "image/jpeg")}
    res = client.post("/upload_register", data={"name": "Ghost"}, files=files)
    assert res.status_code == 400
    assert "No face detected" in res.json()["error"]


def test_upload_register_success(client, sample_frame):
    """Verify POST /upload_register successfully saves image and inserts DB record."""
    user_name = "UploadedSubject"
    created_file = main.STATIC_DIR / f"{user_name}.jpg"

    try:
        _, encoded = cv2.imencode(".jpg", sample_frame)
        files = {"file": (f"{user_name}.jpg", encoded.tobytes(), "image/jpeg")}
        res = client.post("/upload_register", data={"name": user_name}, files=files)
        assert res.status_code == 200
        data = res.json()
        assert data["name"] == user_name
        assert data["faces_detected"] >= 1
        assert "user_id" in data
        assert created_file.exists()

        # Verify DB
        with main.get_db() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (data["user_id"],)).fetchone()
            assert row is not None
            assert row["name"] == user_name
    finally:
        if created_file.exists():
            created_file.unlink()


def test_match_live_face_analysis_error(client, sample_frame):
    """Verify POST /match_live_face handles live face analysis error gracefully."""
    with main.get_db() as conn:
        conn.execute("INSERT INTO users (name, image_path) VALUES (?, ?)", ("Keanu", "static/john wick.jpg"))
        conn.commit()

    with patch.object(camera_manager, "is_active", return_value=True), \
         patch.object(camera_manager, "capture_frame", return_value=sample_frame.copy()), \
         patch("main.perform_face_analysis", side_effect=RuntimeError("Analysis crash")):
        res = client.post("/match_live_face")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "match"
        assert data["name"] == "Keanu"


def test_detect_upload_analysis_error(client, sample_frame):
    """Verify POST /detect_upload handles face analysis error gracefully."""
    _, encoded = cv2.imencode(".jpg", sample_frame)
    files = {"file": ("face.jpg", encoded.tobytes(), "image/jpeg")}

    with patch("main.perform_face_analysis", side_effect=RuntimeError("Upload analysis crash")):
        res = client.post("/detect_upload", files=files)
        assert res.status_code == 200
        data = res.json()
        assert data["faces_detected"] >= 1
        assert "annotated_image" in data


def test_upload_register_special_chars_name(client, sample_frame):
    """Verify POST /upload_register rejects name with only special characters."""
    _, encoded = cv2.imencode(".jpg", sample_frame)
    files = {"file": ("face.jpg", encoded.tobytes(), "image/jpeg")}
    res = client.post("/upload_register", data={"name": "!@#$%^&*()"}, files=files)
    assert res.status_code == 400
    assert "Valid name is required" in res.json()["error"]



