import os
import sys
from typing import Dict, Any

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

        # e.g. {0: "Female", 1: "Male"}
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def predict_gender(self, img: Image.Image) -> Dict[str, Any]:
        """
        img: PIL RGB image.

        Returns:
            {
                "pred_label": "male"/"female",
                "raw": {...}
            }
        """
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
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