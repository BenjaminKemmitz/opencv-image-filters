import cv2
import numpy as np

def morphology(image):
    """
    Morphological opening (noise removal).
    Input: BGR image
    Output: BGR image
    """
    if image is None:
        raise ValueError("Input image is None")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to binary
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)

    # Apply opening (most common morphology op)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Convert back to BGR
    return cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)
