# Canny Edge Detector A multi-stage edge detection algorithm that uses gradients, smoothing, and thresholding to extract clean, accurate edges.

import cv2

def canny_edge(image, low=100, high=200):
    """
    Apply Canny edge detection.
    Input: BGR image (H, W, 3)
    Output: BGR image (H, W, 3)
    """
    if image is None:
        raise ValueError("Input image is None")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny
    edges = cv2.Canny(gray, low, high)

    # Convert back to BGR for pipeline consistency
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
