import numpy as np
import cv2
from PIL import Image

def normalize_skintone(img: Image.Image) -> Image.Image:
    """
    Normalize skin-tone brightness in a gentle, stable way.

    1. Apply CLAHE to equalize local contrast.
    2. Compute median intensity in the central region (proxy for skin).
    3. Apply a *clipped* global scaling factor so that median
       moves toward a target gray level, but never by more than ~30–40%.

    This strongly reduces large brightness differences between skin tones
    without blowing out highlights.
    """
    # PIL -> numpy RGB
    img_np = np.array(img.convert("RGB"))

    # Grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Step 1: CLAHE for local contrast normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)

    h, w = eq.shape

    # Central region as a proxy for facial skin (images are headshots)
    x0, x1 = int(0.25 * w), int(0.75 * w)
    y0, y1 = int(0.25 * h), int(0.75 * h)
    center = eq[y0:y1, x0:x1]

    m = float(np.median(center))
    if m < 1.0:
        m = 1.0

    # Target gray level (mid-gray, not too bright)
    target = 125.0

    # Raw scale factor
    scale = target / m

    # Clip scale so we never over-brighten/over-darken too much
    # e.g. at most +40% brighter or -30% darker
    scale = np.clip(scale, 0.7, 1.4)

    # Apply scaling to the equalized image
    norm = eq.astype(np.float32) * scale
    norm = np.clip(norm, 0, 255).astype(np.uint8)

    # Back to 3-channel RGB
    norm_rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(norm_rgb)