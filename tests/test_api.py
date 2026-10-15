from app import database
from tests.conftest import encode_jpeg, identity_image


def upload(client, path, person, name=None):
    data = {"name": name} if name else None
    return client.post(path, data=data, files={"file": ("p.jpg", encode_jpeg(identity_image(person)), "image/jpeg")})


def register(client, name, person):
    resp = upload(client, "/upload_register", person, name)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_index_and_status(client):
    assert client.get("/").status_code == 200
    body = client.get("/api/status").json()
    assert body["status"] == "online"
    assert body["registered_users_count"] == 0
    assert body["detector"] == "Stub"
    assert body["recognizer_loaded"] is True


def test_register_then_list_and_delete(client, settings):
    user = register(client, "Ann", 1)
    assert user["recognition_ready"] is True
    assert [u["name"] for u in client.get("/users").json()["users"]] == ["Ann"]

    photo = settings.static_dir / user["file_path"].removeprefix("static/")
    assert photo.exists()

    assert client.delete(f"/users/{user['user_id']}").json()["status"] == "user deleted"
    assert not photo.exists()
    assert client.delete(f"/users/{user['user_id']}").status_code == 404


def test_register_rejects_bad_input(client, detector):
    assert upload(client, "/upload_register", 1, "!!!").json() == {"error": "Valid name is required"}

    bad = client.post("/upload_register", data={"name": "x"}, files={"file": ("p.jpg", b"junk", "image/jpeg")})
    assert bad.status_code == 400

    detector.face_count = 0
    assert "No face" in upload(client, "/upload_register", 1, "x").json()["error"]

    detector.face_count = 2
    assert "exactly one face" in upload(client, "/upload_register", 1, "x").json()["error"]

    detector.face_count, detector.face_width = 1, 20
    assert "too small" in upload(client, "/upload_register", 1, "x").json()["error"]
    assert database.count_users() == 0


def test_detect_upload_matches_registered_person(client):
    register(client, "Ann", 1)
    register(client, "Bob", 2)

    result = upload(client, "/detect_upload", 2).json()

    assert result["status"] == "match"
    assert result["name"] == "Bob"
    assert result["faces_detected"] == 1
    assert result["confidence"] == "100.0%"
    assert result["annotated_image"].startswith("static/tmp/")


def test_detect_upload_unknown_person_is_not_matched(client):
    register(client, "Ann", 1)

    result = upload(client, "/detect_upload", 3).json()

    assert result["status"] == "no match"
    assert result["users_count"] == 1
    assert "name" not in result


def test_detect_upload_with_empty_database(client):
    result = upload(client, "/detect_upload", 1).json()
    assert result["status"] == "no match"
    assert result["users_count"] == 0


def test_detect_upload_without_face(client, detector):
    detector.face_count = 0
    result = upload(client, "/detect_upload", 1).json()
    assert result["status"] == "no_face_detected"
    assert result["faces_detected"] == 0


def test_detect_upload_invalid_file(client):
    resp = client.post("/detect_upload", files={"file": ("p.jpg", b"junk", "image/jpeg")})
    assert resp.status_code == 400


def test_match_live_face(client):
    register(client, "Ann", 1)  # the mocked camera frame is person 1
    result = client.post("/match_live_face").json()
    assert result["status"] == "match" and result["name"] == "Ann"
    assert result["live_image"].startswith("static/tmp/")


def test_match_live_face_camera_failure(client, camera):
    camera.capture_frame.return_value = None
    assert client.post("/match_live_face").status_code == 400


def test_capture_photo_registers_user(client):
    resp = client.post("/capture_photo", data={"name": "Cam User"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "Cam User"
    assert database.count_users() == 1


def test_capture_photo_requires_name_and_camera(client, camera):
    assert client.post("/capture_photo", data={"name": "***"}).status_code == 400
    camera.is_active.return_value = False
    camera.start.return_value = False
    assert client.post("/capture_photo", data={"name": "ok"}).status_code == 400


def test_analyze_photo_reports_unavailable_attributes_honestly(client, settings):
    user = register(client, "Ann", 1)
    resp = client.post("/analyze_photo", data={"image_path": user["file_path"]})
    analysis = resp.json()["analysis"]
    assert resp.status_code == 200
    assert analysis["faces_detected"] == 1
    # No DeepFace in the test environment: attributes must be None, never invented.
    assert analysis["age"] is None and analysis["gender"] is None


def test_analyze_photo_rejects_traversal_and_missing(client):
    assert client.post("/analyze_photo", data={"image_path": "../../etc/passwd"}).status_code == 400
    assert client.post("/analyze_photo", data={"image_path": "/etc/passwd"}).status_code == 404  # confined to static/
    assert client.post("/analyze_photo", data={"image_path": "static/nope.jpg"}).status_code == 404


def test_camera_endpoints(client, camera):
    assert client.get("/start_camera").json() == {"status": "camera started"}
    assert client.get("/stop_camera").json() == {"status": "camera stopped"}
    camera.start.return_value = False
    assert client.get("/start_camera").status_code == 500
    camera.is_active.return_value = False
    assert client.get("/video").status_code == 400


def test_video_streams_when_active(client):
    resp = client.get("/video")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("multipart/x-mixed-replace")
