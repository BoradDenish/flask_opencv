"""Download the DeepFace age, gender and emotion weights into models/deepface/ (about 1 GB).

Usage: python scripts/download_deepface_weights.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import analysis  # noqa: E402  (sets DEEPFACE_HOME before deepface loads)


def main() -> None:
    if not analysis.DEEPFACE_AVAILABLE:
        sys.exit("DeepFace is not installed in this environment.")
    for name in ("Age", "Gender", "Emotion"):
        print(f"loading {name} ...", flush=True)
        analysis.DeepFace.build_model(task="facial_attribute", model_name=name)
    print("done")


if __name__ == "__main__":
    main()
