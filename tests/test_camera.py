from unittest.mock import patch, MagicMock
import numpy as np
import pytest
from main import CameraManager


def test_camera_manager_initial_state():
    """Verify initial state of CameraManager instance."""
    cm = CameraManager()
    assert cm.camera is None
    assert cm.is_running is False
    assert cm.is_active() is False


def test_camera_manager_capture_when_inactive():
    """Verify capture_frame returns None when camera is inactive."""
    cm = CameraManager()
    assert cm.capture_frame() is None


def test_camera_manager_stop_when_inactive():
    """Verify stop() returns True safely when camera was not started."""
    cm = CameraManager()
    assert cm.stop() is True
    assert cm.is_active() is False


def test_camera_manager_start_failure():
    """Verify start() returns False when VideoCapture cannot be opened."""
    cm = CameraManager()
    with patch("cv2.VideoCapture") as mock_vc:
        instance = MagicMock()
        instance.isOpened.return_value = False
        mock_vc.return_value = instance

        result = cm.start(camera_index=0)
        assert result is False
        assert cm.is_active() is False
        assert cm.camera is None


def test_camera_manager_start_success(sample_frame):
    """Verify start() returns True and marks camera as active."""
    cm = CameraManager()
    with patch("cv2.VideoCapture") as mock_vc:
        instance = MagicMock()
        instance.isOpened.return_value = True
        instance.read.return_value = (True, sample_frame.copy())
        mock_vc.return_value = instance

        result = cm.start(camera_index=0)
        assert result is True
        assert cm.is_active() is True

        # Calling start again while already active should return True immediately
        assert cm.start(camera_index=0) is True

        cm.stop()
        assert cm.is_active() is False


def test_camera_manager_capture_frame_success(sample_frame):
    """Verify capture_frame returns a copy of the captured frame."""
    cm = CameraManager()
    with patch("cv2.VideoCapture") as mock_vc:
        instance = MagicMock()
        instance.isOpened.return_value = True
        instance.read.return_value = (True, sample_frame.copy())
        mock_vc.return_value = instance

        cm.start(camera_index=0)
        frame = cm.capture_frame()
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == sample_frame.shape

        cm.stop()


def test_camera_manager_capture_frame_read_failure():
    """Verify capture_frame returns None when camera read fails."""
    cm = CameraManager()
    with patch("cv2.VideoCapture") as mock_vc:
        instance = MagicMock()
        instance.isOpened.return_value = True
        instance.read.return_value = (False, None)
        mock_vc.return_value = instance

        cm.start(camera_index=0)
        frame = cm.capture_frame()
        assert frame is None

        cm.stop()


def test_camera_manager_stop_handles_exceptions():
    """Verify stop() handles release() exceptions gracefully."""
    cm = CameraManager()
    with patch("cv2.VideoCapture") as mock_vc:
        instance = MagicMock()
        instance.isOpened.return_value = True
        instance.release.side_effect = RuntimeError("Hardware busy")
        mock_vc.return_value = instance

        cm.start()
        assert cm.stop() is True
        assert cm.camera is None
        assert cm.is_active() is False


def test_camera_manager_gen_frames(sample_frame):
    """Verify gen_frames yields valid multipart JPEG chunks and terminates on stop."""
    cm = CameraManager()
    with patch("cv2.VideoCapture") as mock_vc:
        instance = MagicMock()
        instance.isOpened.return_value = True
        instance.read.return_value = (True, sample_frame.copy())
        mock_vc.return_value = instance

        cm.start()
        gen = cm.gen_frames()

        # Read first chunk
        chunk1 = next(gen)
        assert chunk1.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")

        # Stop camera, next iteration must exit the loop
        cm.stop()
        with pytest.raises(StopIteration):
            next(gen)
