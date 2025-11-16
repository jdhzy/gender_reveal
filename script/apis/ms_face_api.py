# script/apis/ms_face_api.py

import os
import io
from typing import Dict, Any, Optional

import requests
from PIL import Image


class MicrosoftFaceClient:
    """
    Thin wrapper around the Microsoft Face API for gender classification.

    Requires env vars:
        MS_FACE_ENDPOINT  (e.g. https://<region>.api.cognitive.microsoft.com)
        MS_FACE_KEY       (subscription key)

    Uses /face/v1.0/detect with returnFaceAttributes=gender.
    """

    def __init__(self, timeout: float = 5.0):
        endpoint = os.environ.get("MS_FACE_ENDPOINT")
        key = os.environ.get("MS_FACE_KEY")

        if not endpoint or not key:
            raise ValueError("MS_FACE_ENDPOINT and MS_FACE_KEY must be set in the environment.")

        self.endpoint = endpoint.rstrip("/")
        self.key = key
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "microsoft_face"

    def _image_to_bytes(self, img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    def predict_gender(self, img: Image.Image) -> Dict[str, Any]:
        """
        Run gender prediction on a single PIL RGB image.

        Returns:
            {
                "pred_label": "male" / "female" / "unknown",
                "raw": <raw JSON from API or error info>
            }
        """
        url = self.endpoint + "/face/v1.0/detect"

        params = {
            "returnFaceAttributes": "gender",
            "recognitionModel": "recognition_04",
            "returnRecognitionModel": "false",
        }

        headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/octet-stream",
        }

        data = self._image_to_bytes(img)

        try:
            resp = requests.post(
                url,
                params=params,
                headers=headers,
                data=data,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            faces = resp.json()
        except Exception as e:
            return {
                "pred_label": "unknown",
                "raw": {"error": str(e)},
            }

        gender = self._extract_gender(faces)
        return {
            "pred_label": gender,
            "raw": faces,
        }

    def _extract_gender(self, faces_json: Any) -> str:
        """
        Extract 'male'/'female' from API response.
        If multiple faces, choose the largest. If none, 'unknown'.
        """
        if not isinstance(faces_json, list) or len(faces_json) == 0:
            return "unknown"

        def area(face):
            rect = face.get("faceRectangle", {})
            return rect.get("width", 0) * rect.get("height", 0)

        faces_sorted = sorted(faces_json, key=area, reverse=True)
        best = faces_sorted[0]

        attrs: Optional[Dict[str, Any]] = best.get("faceAttributes")
        if not attrs:
            return "unknown"

        gender = str(attrs.get("gender", "")).lower()
        if gender in ("male", "female"):
            return gender
        return "unknown"