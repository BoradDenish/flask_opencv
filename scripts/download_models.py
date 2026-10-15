"""Download the YuNet detector and SFace recognizer ONNX models from the OpenCV Zoo.

Usage: python scripts/download_models.py
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
FILES = {
    "face_detection_yunet_2023mar.onnx": f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def main() -> None:
    models_dir = get_settings().models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        target = models_dir / name
        if target.exists() and target.stat().st_size > 0:
            print(f"ok      {name}")
            continue
        print(f"fetch   {name}")
        partial = target.with_suffix(".part")
        urllib.request.urlretrieve(url, partial)
        partial.replace(target)  # only a complete download ever gets the final name


if __name__ == "__main__":
    main()
