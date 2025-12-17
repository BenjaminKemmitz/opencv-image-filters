#Gaussian Blur Applies a Gaussian kernel to smooth an image, reducing noise and detail while preserving overall structure.

import cv2

def gaussian_blur(image, ksize=(65, 65), sigma=0):
    if image is None:
        raise ValueError("Input image is None")

    return cv2.GaussianBlur(image, ksize, sigma)
