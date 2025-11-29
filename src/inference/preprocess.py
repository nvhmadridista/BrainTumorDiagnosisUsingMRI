from PIL import Image
import numpy as np
import torch

# Mean và std của ImageNet (RGB)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(image: Image.Image) -> torch.Tensor:
    """
    Trả về tensor PyTorch shape (1, 3, 224, 224), float32.
    """

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((224, 224), resample=Image.BILINEAR)
    img = np.array(image).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD

    # HWC -> CHW  
    img = np.transpose(img, (2, 0, 1))  # (3, 224, 224)

    img = np.expand_dims(img, axis=0)   # (1, 3, 224, 224)

    tensor = torch.from_numpy(img).float()

    return tensor

def resize_for_overlay(image: Image.Image) -> Image.Image:
    return image.resize((224, 224), resample=Image.BILINEAR)