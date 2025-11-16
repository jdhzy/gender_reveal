import numpy as np
import cv2
from PIL import Image

def normalize_skintone(img: Image.Image) -> Image.Image:
    """
    Normalize skin-tone brightness using:
    1. central-region median normalization
    2. gamma compression to stabilize highlights

    Returns a 3-channel PIL image.
    """

    # PIL -> numpy RGB
    img_np = np.array(img.convert("RGB"))

    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Central region proxy (middle 50% each direction)
    x0, x1 = int(0.25 * w), int(0.75 * w)
    y0, y1 = int(0.25 * h), int(0.75 * h)
    center = gray[y0:y1, x0:x1]

    m = float(np.median(center))
    if m < 1.0:
        m = 1.0

    # Step 1 — brightness normalization
    target = 140.0  # tune if needed
    scale = target / m

    norm = gray.astype(np.float32) * scale
    norm = np.clip(norm, 0, 255)

    # Step 2 — gamma compression to prevent blown highlights
    # gamma < 1 reduces bright regions
    gamma = 0.8  # you can adjust 0.7–0.9
    norm_scaled = norm / 255.0
    compressed = np.power(norm_scaled, gamma) * 255.0

    compressed = np.clip(compressed, 0, 255).astype(np.uint8)

    # Convert back to 3-channel RGB
    norm_rgb = cv2.cvtColor(compressed, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(norm_rgb)