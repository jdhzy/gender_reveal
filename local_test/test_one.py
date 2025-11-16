import os
import sys
from PIL import Image

# --- Add repo root so Python can find script.data_processing ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)

from script.data_processing.transforms import crop_face, normalize_skintone


def test_image(path, out_prefix):
    img = Image.open(path).convert("RGB")

    cropped = crop_face(img)
    norm = normalize_skintone(img)
    cropped_norm = normalize_skintone(crop_face(img))

    out_dir = os.path.dirname(out_prefix)
    os.makedirs(out_dir, exist_ok=True)

    img.save(f"{out_prefix}_orig.jpg")
    cropped.save(f"{out_prefix}_crop.jpg")
    norm.save(f"{out_prefix}_norm.jpg")
    cropped_norm.save(f"{out_prefix}_crop_norm.jpg")

    print(f"Saved results for: {path}")


def main():
    # EDIT THESE TO YOUR LOCAL TEST IMAGES
    white_face = "local_test/images/WHITE_FACE.jpg"
    black_face = "local_test/images/BLACK_FACE.jpg"

    out_base = "local_test/output"

    test_image(white_face, os.path.join(out_base, "white"))
    test_image(black_face, os.path.join(out_base, "black"))


if __name__ == "__main__":
    main()