import cv2
import numpy as np
from PIL import Image

HAAR_DIR = cv2.data.haarcascades
FACE_CASCADE = cv2.CascadeClassifier(HAAR_DIR + "haarcascade_frontalface_default.xml")

def crop_face(img):
    """
    Takes a PIL image, detects the largest face, and returns a tighter crop
    removing most hair. Same logic as before—just not saving to disk.
    """
    img_np = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return img  # fallback: return original

    # largest face
    faces = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)
    x, y, w, h = faces[0]

    expand = 0.15
    nx = max(0, int(x - w * expand))
    ny = max(0, int(y - h * expand * 0.5))
    nw = min(img_np.shape[1], int(w * (1 + 2*expand)))
    nh = min(img_np.shape[0], int(h * (1 + expand)))

    crop_np = img_np[ny:ny+nh, nx:nx+nw]

    if crop_np.size == 0:
        return img

    return Image.fromarray(crop_np)



def normalize_skintone(img):
    """
    Apply CLAHE to normalize skin tone brightness differences while preserving structure.
    Input: PIL RGB image
    Output: 3-channel PIL RGB image with normalized brightness
    """
    # PIL -> numpy (RGB)
    img_np = np.array(img.convert("RGB"))

    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # CLAHE (best for faces)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(gray)

    # Convert back to 3-channel RGB
    norm_rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)

    # numpy -> PIL
    return Image.fromarray(norm_rgb)