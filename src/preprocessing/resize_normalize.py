import cv2
import numpy as np

def load_and_resize(path, size=224):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return img

# Nomalize pixel values to [0, 1] range
def normalize_minmax(img):
    return img.astype(np.float32) / 255.0

# Standard ImageNet normalization
def normalize_imagenet(img):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = normalize_minmax(img)
    return (img - mean) / std
