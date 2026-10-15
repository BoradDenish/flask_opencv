import numpy as np
import pytest

from app.services.recognition import FaceRecognizer


@pytest.fixture
def users():
    return [{"id": 1, "name": "ann"}, {"id": 2, "name": "bob"}]


def unit(*values):
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_confidence_is_half_at_threshold(recognizer):
    t = recognizer.settings.match_threshold
    assert recognizer.confidence(t) == pytest.approx(0.5)
    assert recognizer.confidence(1.0) == pytest.approx(1.0)
    assert recognizer.confidence(0.0) == 0.0
    assert recognizer.confidence(-0.4) == 0.0


def test_confidence_is_monotonic(recognizer):
    scores = np.linspace(-1, 1, 41)
    values = [recognizer.confidence(float(s)) for s in scores]
    assert values == sorted(values)


def test_best_match_picks_closest_user(recognizer, users):
    matrix = np.stack([unit(1, 0), unit(0, 1)])
    accepted, closest = recognizer.best_match(unit(0.1, 1), users, matrix)
    assert accepted.user["name"] == "bob"
    assert accepted is closest


def test_best_match_rejects_below_threshold(recognizer, users):
    matrix = np.stack([unit(1, 0), unit(0, 1)])
    accepted, closest = recognizer.best_match(unit(1, 1) * -1, users, matrix)
    assert accepted is None
    assert closest.score < recognizer.settings.match_threshold


def test_best_match_without_users(recognizer):
    assert recognizer.best_match(unit(1, 0), [], np.empty((0, 0), dtype=np.float32)) == (None, None)


def test_recognizer_without_model_is_unavailable(settings, tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "none"))
    from app.config import get_settings

    rec = FaceRecognizer(get_settings())
    assert rec.available is False
    with pytest.raises(RuntimeError):
        rec.embed(np.zeros((10, 10, 3), np.uint8), None)
