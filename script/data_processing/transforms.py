import numpy as np
import cv2
from PIL import Image

def normalize_skintone(img: Image.Image) -> Image.Image:
    """
    Normalize skin-tone brightness by forcing the central face region
    to have a fixed median gray level.

    Input:  PIL RGB image
    Output: PIL RGB image (3-channel) with much more similar overall
            face brightness across different skin tones.
    """
    # PIL -> numpy RGB
    img_np = np.array(img.convert("RGB"))

    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Use the central region as a proxy for face skin
    # (FairFace images are mostly centered headshots)
    x0 = int(0.25 * w)
    x1 = int(0.75 * w)
    y0 = int(0.25 * h)
    y1 = int(0.75 * h)
    roi = gray[y0:y1, x0:x1]

    # Robust statistic: median intensity in the central region
    m = float(np.median(roi))
    if m < 1.0:
        m = 1.0  # avoid divide-by-zero

    # Target gray level for skin (some mid-gray)
    target = 140.0  # tweakable: 120–150 is reasonable

    scale = target / m

    # Apply scaling to the whole image
    norm = gray.astype(np.float32) * scale
    norm = np.clip(norm, 0, 255).astype(np.uint8)

    # Back to 3-channel RGB so models expecting RGB still work
    norm_rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(norm_rgb)