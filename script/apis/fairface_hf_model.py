import os
import sys
from typing import Dict, Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

# Resolve project root for consistent imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

# Local model directory (where you scp'ed the HF repo with pytorch_model.bin)
DEFAULT_MODEL_DIR = os.path.join(
    PROJECT_ROOT, "metadata", "models", "fairface_gender_image_detection_pt2"
)


class HFFairFaceGenderModel:
    """
    HuggingFace FairFace gender classifier wrapper.

    Uses a *local* copy of the HF model saved under:
        metadata/models/fairface_gender_image_detection_pt2

    We avoid calling the feature extractor's __call__ directly because on SCC
    (old transformers) its `size` field can be stored in a way that breaks
    PIL.Image.resize(). Instead we:
      - manually resize to a square target size
      - manually normalize using image_mean / image_std from the processor
    """

    def __init__(self, model_dir: str = None, device: str = None):
        self.name = "hf_fairface_gender"

        # Use local path, not remote HF id
        if model_dir is None:
            model_dir = DEFAULT_MODEL_DIR
        self.model_dir = model_dir

        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(
                f"Model directory not found at {self.model_dir}. "
                f"Make sure you copied the HuggingFace model there "
                f"and that it contains pytorch_model.bin."
            )

        print(f"Loading model from local dir: {self.model_dir}")

        # Pick device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load processor + model from local directory (no internet)
        self.processor = AutoFeatureExtractor.from_pretrained(self.model_dir)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()

        # Label mapping, e.g. {0: "Female", 1: "Male"}
        self.id2label = self.model.config.id2label

        # Figure out target size (fallback to 224 if anything weird)
        size = getattr(self.processor, "size", 224)
        target = 224
        if isinstance(size, int):
            target = size
        elif isinstance(size, (list, tuple)) and len(size) >= 1:
            # e.g. [224, 224]
            try:
                target = int(size[0])
            except Exception:
                target = 224
        elif isinstance(size, dict):
            # e.g. {"height": 224, "width": 224}
            h = size.get("height", 224)
            try:
                target = int(h)
            except Exception:
                target = 224
        elif isinstance(size, str):
            # e.g. "224"
            try:
                target = int(size)
            except Exception:
                target = 224

        self.target_size = target

        # Mean / std for normalization
        self.image_mean = getattr(self.processor, "image_mean", [0.5, 0.5, 0.5])
        self.image_std = getattr(self.processor, "image_std", [0.5, 0.5, 0.5])

        # Convert to tensors we can broadcast easily
        self.mean_tensor = torch.tensor(self.image_mean).view(3, 1, 1)
        self.std_tensor = torch.tensor(self.image_std).view(3, 1, 1)

    @torch.no_grad()
    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        """
        Manual preprocessing:
          - ensure RGB
          - resize to (target_size, target_size)
          - convert to float tensor in [0, 1]
          - normalize with model's mean/std

        Returns:
            1 x 3 x H x W tensor on self.device
        """
        img = img.convert("RGB")
        img = img.resize((self.target_size, self.target_size))

        # PIL -> numpy -> torch
        arr = np.array(img).astype("float32") / 255.0  # H x W x C
        arr = np.transpose(arr, (2, 0, 1))  # C x H x W

        tensor = torch.from_numpy(arr)  # 3 x H x W
        # Normalize
        tensor = (tensor - self.mean_tensor) / self.std_tensor
        # Add batch dimension
        tensor = tensor.unsqueeze(0).to(self.device)  # 1 x 3 x H x W
        return tensor

    @torch.no_grad()
    def predict_gender(self, img: Image.Image) -> Dict[str, Any]:
        """
        img: PIL RGB image (or anything PIL can convert).

        Returns:
            {
                "pred_label": "male"/"female",
                "raw": {...}
            }
        """
        pixel_values = self._preprocess(img)

        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits  # 1 x num_labels
        probs = torch.softmax(logits, dim=1)[0]

        pred_idx = int(torch.argmax(probs))
        pred_label = self.id2label[pred_idx]  # e.g. "Female" or "Male"

        return {
            "pred_label": pred_label.lower(),
            "raw": {
                "logits": logits.cpu().numpy().tolist(),
                "probs": probs.cpu().numpy().tolist(),
            },
        }