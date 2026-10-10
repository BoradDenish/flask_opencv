from pathlib import Path
import cv2
import numpy as np
import pytest

import main
from main import (
    extract_face_hist,
    perform_face_analysis,
    perform_face_verification,
    detect_faces_and_annotate,
    match_face_against_db,
)


@pytest.fixture
def sample_face_path():
    """Returns absolute path to sample face image."""
    img_path = main.STATIC_DIR / "john wick.jpg"
    assert img_path.exists(), f"Sample image {img_path} not found"
    return str(img_path)


@pytest.fixture
def blank_image_path(tmp_path):
    """Creates a blank black image with no faces."""
    blank_img = np.zeros((300, 300, 3), dtype=np.uint8)
    p = tmp_path / "blank.jpg"
    cv2.imwrite(str(p), blank_img)
    return str(p)


def test_extract_face_hist_with_no_face(blank_image_path):
    """Verify extract_face_hist returns None when no face is found in image."""
    img = cv2.imread(blank_image_path)
    hist = extract_face_hist(img)
    assert hist is None


def test_extract_face_hist_with_valid_face(sample_face_path):
    """Verify extract_face_hist returns normalized histogram for image with face."""
    img = cv2.imread(sample_face_path)
    hist = extract_face_hist(img)
    assert hist is not None
    assert isinstance(hist, np.ndarray)
    assert hist.shape == (256, 1)
    # Normalized between 0 and 1
    assert hist.min() >= 0.0
    assert hist.max() <= 1.0


def test_extract_face_hist_when_cascade_none(sample_face_path, monkeypatch):
    """Verify extract_face_hist returns None if face_cascade is not loaded."""
    monkeypatch.setattr(main, "face_cascade", None)
    img = cv2.imread(sample_face_path)
    hist = extract_face_hist(img)
    assert hist is None


def test_perform_face_analysis_file_not_found():
    """Verify perform_face_analysis raises FileNotFoundError for missing images."""
    with pytest.raises(FileNotFoundError):
        perform_face_analysis("non_existent_path_xyz123.jpg")


def test_perform_face_analysis_valid_face(sample_face_path):
    """Verify perform_face_analysis returns required schema for face image."""
    result = perform_face_analysis(sample_face_path)
    assert isinstance(result, dict)
    assert "age" in result
    assert "gender" in result
    assert "emotion" in result
    assert "engine" in result


def test_perform_face_analysis_blank_image(blank_image_path):
    """Verify perform_face_analysis handles images with no faces."""
    result = perform_face_analysis(blank_image_path)
    assert isinstance(result, dict)
    assert result.get("faces_detected") == 0


def test_perform_face_verification_nonexistent_files(sample_face_path):
    """Verify verification returns False if either image file does not exist."""
    res1 = perform_face_verification("missing_1.jpg", sample_face_path)
    assert res1["verified"] is False

    res2 = perform_face_verification(sample_face_path, "missing_2.jpg")
    assert res2["verified"] is False


def test_perform_face_verification_same_image(sample_face_path):
    """Verify identical face images result in verified == True."""
    res = perform_face_verification(sample_face_path, sample_face_path)
    assert isinstance(res, dict)
    assert res.get("verified") is True
    assert res.get("distance") == 0.0 or res.get("distance", 1.0) < 0.3


def test_perform_face_verification_face_vs_blank(sample_face_path, blank_image_path):
    """Verify comparing a face to a blank image results in verified == False."""
    res = perform_face_verification(sample_face_path, blank_image_path)
    assert res.get("verified") is False


def test_perform_face_analysis_with_deepface(sample_face_path, monkeypatch):
    """Verify perform_face_analysis behavior when DeepFace is available."""
    from unittest.mock import MagicMock
    mock_deepface = MagicMock()
    mock_deepface.analyze.return_value = [{
        "age": 30,
        "dominant_gender": "Woman",
        "dominant_emotion": "happy",
        "emotion": {"happy": 98.5, "neutral": 1.5}
    }]
    monkeypatch.setattr(main, "DEEPFACE_AVAILABLE", True)
    monkeypatch.setattr(main, "DeepFace", mock_deepface, raising=False)

    res = perform_face_analysis(sample_face_path)
    assert res["engine"] == "DeepFace"
    assert res["age"] == 30
    assert res["gender"] == "Woman"
    assert res["emotion"]["dominant_emotion"] == "happy"


def test_perform_face_analysis_with_deepface_dict_gender(sample_face_path, monkeypatch):
    """Verify perform_face_analysis extracts dominant gender from dictionary."""
    from unittest.mock import MagicMock
    mock_deepface = MagicMock()
    mock_deepface.analyze.return_value = {
        "age": 25,
        "gender": {"Man": 90.0, "Woman": 10.0},
        "emotion": {"neutral": 95.0}
    }
    monkeypatch.setattr(main, "DEEPFACE_AVAILABLE", True)
    monkeypatch.setattr(main, "DeepFace", mock_deepface, raising=False)

    res = perform_face_analysis(sample_face_path)
    assert res["engine"] == "DeepFace"
    assert res["age"] == 25
    assert res["gender"] == "Man"
    assert res["emotion"]["dominant_emotion"] == "neutral"


def test_perform_face_verification_with_deepface(sample_face_path, monkeypatch):
    """Verify perform_face_verification behavior when DeepFace is available."""
    from unittest.mock import MagicMock
    mock_deepface = MagicMock()
    mock_deepface.verify.return_value = {
        "verified": True,
        "distance": 0.12,
        "threshold": 0.40
    }
    monkeypatch.setattr(main, "DEEPFACE_AVAILABLE", True)
    monkeypatch.setattr(main, "DeepFace", mock_deepface, raising=False)

    res = perform_face_verification(sample_face_path, sample_face_path)
    assert res["engine"] == "DeepFace"
    assert res["verified"] is True
    assert res["distance"] == 0.12
    assert res["threshold"] == 0.40


def test_detect_faces_and_annotate_blank(blank_image_path):
    """Verify detect_faces_and_annotate detects 0 faces on blank image."""
    img = cv2.imread(blank_image_path)
    annotated, faces = detect_faces_and_annotate(img)
    assert len(faces) == 0
    assert annotated.shape == img.shape


def test_detect_faces_and_annotate_valid_face(sample_face_path):
    """Verify detect_faces_and_annotate detects faces and returns coordinates."""
    img = cv2.imread(sample_face_path)
    annotated, faces = detect_faces_and_annotate(img)
    assert len(faces) >= 1
    face0 = faces[0]
    assert "x" in face0 and "y" in face0 and "w" in face0 and "h" in face0
    assert face0["w"] > 0 and face0["h"] > 0


def test_detect_faces_and_annotate_no_cascade(sample_face_path, monkeypatch):
    """Verify detect_faces_and_annotate returns empty list when cascade is None."""
    monkeypatch.setattr(main, "face_cascade", None)
    img = cv2.imread(sample_face_path)
    annotated, faces = detect_faces_and_annotate(img)
    assert len(faces) == 0


def test_match_face_against_db_empty(isolated_db, sample_face_path):
    """Verify match_face_against_db returns no match when database is empty."""
    res = match_face_against_db(sample_face_path)
    assert res["status"] == "no match"
    assert res["users_count"] == 0


def test_match_face_against_db_success(isolated_db, sample_face_path):
    """Verify match_face_against_db identifies registered matching user."""
    with main.get_db() as conn:
        conn.execute("INSERT INTO users (name, image_path) VALUES (?, ?)", ("Keanu DB", "static/john wick.jpg"))
        conn.commit()

    res = match_face_against_db(sample_face_path)
    assert res["status"] == "match"
    assert res["name"] == "Keanu DB"
    assert "confidence" in res
    assert res["matched_image"] == "static/john wick.jpg"


def test_match_face_against_db_missing_file(isolated_db, sample_face_path):
    """Verify match_face_against_db handles missing candidate image paths gracefully."""
    with main.get_db() as conn:
        conn.execute("INSERT INTO users (name, image_path) VALUES (?, ?)", ("Missing", "static/not_found_123.jpg"))
        conn.commit()

    res = match_face_against_db(sample_face_path)
    assert res["status"] == "no match"


def test_match_face_against_db_no_match(isolated_db, sample_face_path, blank_image_path):
    """Verify match_face_against_db reports no match when comparing different faces."""
    with main.get_db() as conn:
        conn.execute("INSERT INTO users (name, image_path) VALUES (?, ?)", ("Blank Person", "static/blank_img_test.jpg"))
        conn.commit()

    # Save blank image to candidate location
    cand_path = main.STATIC_DIR / "blank_img_test.jpg"
    shutil_copy = cv2.imwrite(str(cand_path), cv2.imread(blank_image_path))
    try:
        res = match_face_against_db(sample_face_path)
        assert res["status"] == "no match"
    finally:
        if cand_path.exists():
            cand_path.unlink()


def test_match_face_against_db_exception(isolated_db, sample_face_path, monkeypatch):
    """Verify match_face_against_db continues cleanly when verification raises an error."""
    with main.get_db() as conn:
        conn.execute("INSERT INTO users (name, image_path) VALUES (?, ?)", ("Faulty User", "static/john wick.jpg"))
        conn.commit()

    def _mock_verify(*args, **kwargs):
        raise RuntimeError("Corrupt image data")

    monkeypatch.setattr(main, "perform_face_verification", _mock_verify)
    res = match_face_against_db(sample_face_path)
    assert res["status"] == "no match"


