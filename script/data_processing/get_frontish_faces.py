import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import mediapipe as mp
import cv2
import numpy as np

# ------------ SETUP ----------------

FAIRFACE_DIR = "../../data/fairface"
OUT_DIR = "../../data/fairface_frontish"

os.makedirs(OUT_DIR, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE_LANDMARKS = [33, 133]
RIGHT_EYE_LANDMARKS = [362, 263]


# ------------ HELPER FUNCTION ----------------

def both_eyes_detected(image):
    """
    Returns True if both eyes are detected with MediaPipe FaceMesh.
    Returns False otherwise.
    """

    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True
    ) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return False

        landmarks = results.multi_face_landmarks[0]

        try:
            left_eye = [landmarks.landmark[i] for i in LEFT_EYE_LANDMARKS]
            right_eye = [landmarks.landmark[i] for i in RIGHT_EYE_LANDMARKS]

            # Check if landmark coordinates exist
            if any(l.x == 0 and l.y == 0 for l in left_eye):
                return False
            if any(l.x == 0 and l.y == 0 for l in right_eye):
                return False

            return True

        except:
            return False


# ------------ MAIN CLEANING FUNCTION ----------------

def clean_split(split):
    split_in = os.path.join(FAIRFACE_DIR, split)
    split_out = os.path.join(OUT_DIR, split)
    os.makedirs(split_out, exist_ok=True)

    csv_in = os.path.join(split_in, "labels.csv")
    df = pd.read_csv(csv_in)

    kept_rows = []

    print(f"Processing {split}…")
    for _, row in tqdm(df.iterrows(), total=len(df)):

        img_path = os.path.join(split_in, row["filename"])

        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue

        # *** CORE RULE: BOTH EYES MUST BE VISIBLE ***
        if not both_eyes_detected(img):
            continue

        # Save filtered image
        out_path = os.path.join(split_out, row["filename"])
        img.save(out_path)

        kept_rows.append(row)

    # Save new CSV
    out_csv = os.path.join(split_out, "labels.csv")
    pd.DataFrame(kept_rows).to_csv(out_csv, index=False)

    print(f"Finished {split}: kept {len(kept_rows)} / {len(df)} images")


# ------------ MAIN ENTRY POINT ----------------

def main():
    for split in ["train", "validation", "test"]:
        clean_split(split)

    print("All splits processed. Frontish dataset saved to:", OUT_DIR)


if __name__ == "__main__":
    main()