import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Runtime configuration, read from the environment each time it is requested."""

    base_dir: Path
    static_dir: Path
    templates_dir: Path
    haar_dir: Path
    models_dir: Path
    db_file: Path
    camera_index: int
    # Cosine similarity at or above this value counts as the same person.
    # 0.363 is the threshold published by the OpenCV Zoo for SFace.
    match_threshold: float
    # Minimum YuNet confidence for a detection to be kept.
    detect_score: float
    # Smallest face (in pixels, width) accepted when registering a user.
    min_face_size: int

    @property
    def yunet_path(self) -> Path:
        return self.models_dir / "face_detection_yunet_2023mar.onnx"

    @property
    def sface_path(self) -> Path:
        return self.models_dir / "face_recognition_sface_2021dec.onnx"

    @property
    def tmp_dir(self) -> Path:
        return self.static_dir / "tmp"


def get_settings() -> Settings:
    static_dir = Path(os.getenv("STATIC_DIR", str(BASE_DIR / "static")))
    return Settings(
        base_dir=BASE_DIR,
        static_dir=static_dir,
        templates_dir=BASE_DIR / "templates",
        haar_dir=BASE_DIR / "Haarcascades",
        models_dir=Path(os.getenv("MODELS_DIR", str(BASE_DIR / "models"))),
        db_file=Path(os.getenv("DB_FILE", str(BASE_DIR / "users.db"))),
        camera_index=int(os.getenv("CAMERA_INDEX", "0")),
        match_threshold=float(os.getenv("MATCH_THRESHOLD", "0.363")),
        detect_score=float(os.getenv("DETECT_SCORE", "0.9")),
        min_face_size=int(os.getenv("MIN_FACE_SIZE", "60")),
    )
