import os
import sys
from PIL import Image

# --- Add project root to Python path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

from script.apis.ms_face_api import MicrosoftFaceClient


def main():
    # Pick any mini-eval image
    img_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "mini_eval",
        "validation",
        "0000289.jpg"  # <-- EDIT this
    )

    img = Image.open(img_path).convert("RGB")

    client = MicrosoftFaceClient()
    result = client.predict_gender(img)
    print("Result:", result)


if __name__ == "__main__":
    main()