import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import mediapipe as mp
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
# MediaPipe FaceMesh setup
# -------------------------
mp_face_mesh = mp.solutions.face_mesh
# Use two stable corner landmarks for each eye
LEFT_EYE_LANDMARKS = [33, 133]
RIGHT_EYE_LANDMARKS = [362, 263]


def both_eyes_detected(image):
    """
    Returns True if both eyes are detected with MediaPipe FaceMesh.
    Returns False otherwise.
    """
    # PIL -> OpenCV BGR
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True
    ) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return False

        landmarks = results.multi_face_landmarks[0]

        try:
            left_eye = [landmarks.landmark[i] for i in LEFT_EYE_LANDMARKS]
            right_eye = [landmarks.landmark[i] for i in RIGHT_EYE_LANDMARKS]

            # Basic existence check – if they collapse to (0, 0) it's junk
            if any((l.x == 0 and l.y == 0) for l in left_eye):
                return False
            if any((l.x == 0 and l.y == 0) for l in right_eye):
                return False

            return True

        except Exception:
            return False


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
            # skip unreadable images
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
    # Right now you have train + validation; add "test" later if you download it
    for split in ["train", "validation"]:
        split_path = os.path.join(FAIRFACE_DIR, split)
        if not os.path.exists(split_path):
            print(f"Skip {split}: {split_path} does not exist.")
            continue
        clean_split(split)

    print("All available splits processed. Front-ish dataset saved to:", OUT_BASE_DIR)


if __name__ == "__main__":
    main()