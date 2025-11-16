import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np

# -------------------------
# Path setup (based on this file's location)
# -------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

FAIRFACE_DIR = os.path.join(PROJECT_ROOT, "data", "fairface")
OUT_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "cleaned", "frontish")

os.makedirs(OUT_BASE_DIR, exist_ok=True)

# -------------------------
# OpenCV Haar Cascades setup
# -------------------------
HAAR_DIR = cv2.data.haarcascades
FACE_CASCADE = cv2.CascadeClassifier(os.path.join(HAAR_DIR, "haarcascade_frontalface_default.xml"))
EYE_CASCADE = cv2.CascadeClassifier(os.path.join(HAAR_DIR, "haarcascade_eye.xml"))


def both_eyes_detected(image):
    """
    Returns True if a frontal-ish face is detected and at least 2 eyes are found
    inside the face bounding box. Returns False otherwise.
    """
    # PIL -> OpenCV grayscale
    img_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return False

    # Take the largest face
    faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
    x, y, w, h = faces[0]

    roi_gray = gray[y:y + h, x:x + w]

    # Detect eyes within the face region
    eyes = EYE_CASCADE.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)

    # If at least 2 eyes are detected, we consider the face "front-ish"
    return len(eyes) >= 2


def clean_split(split):
    split_in = os.path.join(FAIRFACE_DIR, split)
    split_out = os.path.join(OUT_BASE_DIR, split)
    os.makedirs(split_out, exist_ok=True)

    csv_in = os.path.join(split_in, "labels.csv")
    df = pd.read_csv(csv_in)

    kept_rows = []

    print(f"Processing {split}…")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(split_in, row["filename"])

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # Skip unreadable images
            continue

        # CORE RULE: both eyes must be detected → "front-ish"
        if not both_eyes_detected(img):
            continue

        out_path = os.path.join(split_out, row["filename"])
        img.save(out_path)
        kept_rows.append(row)

    out_csv = os.path.join(split_out, "labels.csv")
    pd.DataFrame(kept_rows).to_csv(out_csv, index=False)

    print(f"Finished {split}: kept {len(kept_rows)} / {len(df)} images")


def main():
    for split in ["train", "validation"]:
        split_path = os.path.join(FAIRFACE_DIR, split)
        if not os.path.exists(split_path):
            print(f"Skip {split}: {split_path} does not exist.")
            continue
        clean_split(split)

    print("All available splits processed. Front-ish dataset saved to:", OUT_BASE_DIR)


if __name__ == "__main__":
    main()