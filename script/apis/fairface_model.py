import os
import sys
from typing import Dict, Any

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

# Optional: if you later clone the official FairFace repo,
# you can import their exact model architecture.
# For now we'll use a standard ResNet18 and a checkpoint path placeholder.


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Where we expect the pre-trained weights to live.
# You'll download / place a FairFace gender model here.
DEFAULT_WEIGHTS_PATH = os.path.join(
    PROJECT_ROOT, "metadata", "models", "fairface_gender_resnet18.pth"
)


class SimpleResNet18Gender(nn.Module):
    """
    Minimal ResNet18-based classifier for 2 gender classes.
    We assume the checkpoint was trained with this structure:
        backbone (ResNet18) + final fc -> 2 outputs (Male/Female).
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        from torchvision.models import resnet18

        self.backbone = resnet18(pretrained=False)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class FairFaceGenderModel:
    """
    Local FairFace-based gender classifier.

    Exposes a simple .predict_gender(pil_img) -> {"pred_label": ..., "raw": ...}
    interface, similar to the MicrosoftFaceClient.
    """

    def __init__(self, weights_path: str = DEFAULT_WEIGHTS_PATH, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = SimpleResNet18Gender(num_classes=2)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"FairFace weights not found at {weights_path}. "
                f"Please download a gender model checkpoint and place it there."
            )

        state = torch.load(weights_path, map_location=self.device)
        # Some checkpoints store under 'state_dict'
        if isinstance(state, dict) and "state_dict" in state:
            state = {k.replace("module.", "").replace("backbone.", ""): v
                     for k, v in state["state_dict"].items()}
            self.model.load_state_dict(state, strict=False)
        else:
            self.model.load_state_dict(state, strict=False)

        self.model.to(self.device)
        self.model.eval()

        # Standard FairFace-style transforms
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # We’ll assume logits index 0 = Male, 1 = Female
        self.idx_to_gender = {0: "Male", 1: "Female"}

    @property
    def name(self) -> str:
        return "fairface_resnet18_gender"

    @torch.no_grad()
    def predict_gender(self, img: Image.Image) -> Dict[str, Any]:
        """
        img: PIL RGB image.
        """
        x = self.transform(img).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        pred_label = self.idx_to_gender.get(pred_idx, "Unknown")

        return {
            "pred_label": pred_label.lower(),  # 'male'/'female'
            "raw": {
                "logits": logits.cpu().numpy().tolist(),
                "probs": probs.cpu().numpy().tolist(),
            },
        }