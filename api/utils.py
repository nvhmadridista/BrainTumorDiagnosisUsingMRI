from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import List, Tuple
import base64

import torch
from PIL import Image
from torchvision import transforms


DEFAULT_SIZE: Tuple[int, int] = (224, 224)
CLASS_NAMES: List[str] = ["glioma", "meningioma", "notumor", "pituitary"]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_preprocess() -> transforms.Compose:
    """
    Build preprocessing matching training pipeline:
    - Resize to 224x224
    - ToTensor (scales to [0,1])
    - No normalization (to match training)
    """
    return transforms.Compose([
        transforms.Resize(DEFAULT_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])


def load_image_from_bytes(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data)).convert("RGB")


def pil_to_base64_png(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")