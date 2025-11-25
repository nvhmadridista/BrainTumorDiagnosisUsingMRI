from PIL import Image
import numpy as np

# Mean và std của ImageNet (RGB)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def resize_and_center_crop(image: Image.Image, size=224, resize_size=256) -> Image.Image:
    """
    Resize ảnh về resize_size rồi center crop size x size.
    """
    image = image.resize((resize_size, resize_size), resample=Image.BILINEAR)
    left = (resize_size - size) // 2
    top = (resize_size - size) // 2
    right = left + size
    bottom = top + size
    image = image.crop((left, top, right, bottom))
    return image

def preprocess(image: Image.Image) -> np.ndarray:
    """
    Tiền xử lý ảnh PIL thành input numpy float32 cho model.
    Output shape: (1, 224, 224, 3), dtype=float32
    """

    img = resize_and_center_crop(image)
    img = np.array(img).astype(np.float32) / 255.0

    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    elif img.shape[2] == 1:
        img = np.concatenate([img]*3, axis=-1)

    img = (img - IMAGENET_MEAN) / IMAGENET_STD

    img = np.expand_dims(img, axis=0)

    return img
