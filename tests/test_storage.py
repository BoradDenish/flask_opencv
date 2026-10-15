import pytest

from app.errors import ApiError
from app.services import storage


def test_clean_name_strips_unsafe_characters():
    assert storage.clean_name("  Jo<script>hn_Doe-2  ") == "Joscripthn_Doe-2"


@pytest.mark.parametrize("raw", ["", "   ", "!!!", "../"])
def test_clean_name_rejects_empty_result(raw):
    with pytest.raises(ApiError) as exc:
        storage.clean_name(raw)
    assert exc.value.status_code == 400


def test_decode_image_rejects_garbage_and_empty():
    for payload in (b"", b"not an image"):
        with pytest.raises(ApiError) as exc:
            storage.decode_image(payload)
        assert exc.value.status_code == 400


@pytest.mark.parametrize("bad", ["../secret.jpg", "static/../../x.jpg", "..\\x.jpg"])
def test_resolve_static_path_blocks_traversal(settings, bad):
    with pytest.raises(ApiError):
        storage.resolve_static_path(settings, bad)


def test_resolve_static_path_accepts_static_prefix(settings):
    settings.static_dir.mkdir(parents=True, exist_ok=True)
    path = storage.resolve_static_path(settings, "static/a.jpg")
    assert path == settings.static_dir.resolve() / "a.jpg"


def test_same_name_does_not_overwrite_photo(settings, sample_frame):
    one = storage.save_user_photo(settings, "ann", sample_frame)
    two = storage.save_user_photo(settings, "ann", sample_frame)
    assert one != two
    assert storage.resolve_static_path(settings, one).exists()
    assert storage.resolve_static_path(settings, two).exists()
