import numpy as np
from PIL import Image


def regular_normalize_for_vis(
    img: Image.Image,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    clip_sigma: float = 2.0,
) -> Image.Image:
    """
    Apply a standard per-channel mean/std normalization and
    then rescale back to an 8-bit RGB image for visualization.

    This is meant as a baseline "regular" normalization to compare
    against `normalize_skintone`:
      - convert to float in [0, 1]
      - normalize using fixed mean/std (like HF models)
      - optionally clip to a limited sigma range
      - linearly map the result into [0, 255] for display
    """
    # PIL -> numpy RGB in [0, 1]
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0  # (H, W, C)

    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    # Standard per-channel normalization
    norm = (arr - mean_arr) / (std_arr + 1e-6)

    # Optional clipping in "sigma space" for a stable dynamic range
    if clip_sigma is not None:
        norm = np.clip(norm, -clip_sigma, clip_sigma)

    # Map normalized values back to [0, 1] for visualization
    norm_min = float(norm.min())
    norm_max = float(norm.max())
    if norm_max <= norm_min + 1e-6:
        vis = np.zeros_like(norm)
    else:
        vis = (norm - norm_min) / (norm_max - norm_min)

    # Back to uint8 RGB
    vis_uint8 = (vis * 255.0).astype(np.uint8)
    return Image.fromarray(vis_uint8)


__all__ = ["regular_normalize_for_vis"]

