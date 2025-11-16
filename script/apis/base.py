# script/apis/base.py

from abc import ABC, abstractmethod
from PIL import Image
from typing import Dict, Any


class GenderAPIClient(ABC):
    """
    Abstract base class for any external gender-classification API.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name for logging/results (e.g., 'microsoft_face')."""
        pass

    @abstractmethod
    def predict_gender(self, img: Image.Image) -> Dict[str, Any]:
        """
        Run gender prediction on a single image.

        Returns a dict with at least:
            {
                "pred_label": "male" / "female" / "unknown",
                "raw": <raw_response_object or json>
            }
        """
        pass