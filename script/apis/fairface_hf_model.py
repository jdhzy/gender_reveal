import os
import sys
from typing import Dict, Any

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Resolve project root for consistent imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)


class HFFairFaceGenderModel:
    """
    HuggingFace FairFace gender classifier wrapper.

    Model: dima806/fairface_gender_image_detection
    Labels (from config): {0: 'Female', 1: 'Male'}
    """

    def __init__(self, device: str = None):
        self.name = "hf_fairface_gender"

        self.model_id = "dima806/fairface_gender_image_detection"
        print(f"Loading model: {self.model_id}")

        # Pick device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load processor + model
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label  # e.g. {0: 'Female', 1: 'Male'}

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
        pred_label = self.id2label[pred_idx]  # "Female" or "Male"

        return {
            "pred_label": pred_label.lower(),
            "raw": {
                "logits": logits.cpu().numpy().tolist(),
                "probs": probs.cpu().numpy().tolist(),
            },
        }
