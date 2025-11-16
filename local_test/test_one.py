import os
import sys
from PIL import Image

# --- Add repo root so we can import script.data_processing ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)

from script.data_processing.transforms import normalize_skintone


def test_image(path, out_prefix):
    img = Image.open(path).convert("RGB")

    norm = normalize_skintone(img)

    out_dir = os.path.dirname(out_prefix)
    os.makedirs(out_dir, exist_ok=True)

    img.save(f"{out_prefix}_orig.jpg")
    norm.save(f"{out_prefix}_norm.jpg")

    print(f"Saved results for: {path}")


def main():
    # EDIT THESE TWO PATHS TO YOUR TEST IMAGES
    white_face = "local_test/images/white.jpg"
    black_face = "local_test/images/black.jpg"

    out_base = "local_test/output"

    test_image(white_face, os.path.join(out_base, "white"))
    test_image(black_face, os.path.join(out_base, "black"))


if __name__ == "__main__":
    main()