import cv2

def remove_noise(img, method="bilateral"):
    if method == "gaussian":
        return cv2.GaussianBlur(img, (3,3), 0)
    elif method == "median":
        return cv2.medianBlur(img, 3)
    else:
        return cv2.bilateralFilter(img, 9, 75, 75)
