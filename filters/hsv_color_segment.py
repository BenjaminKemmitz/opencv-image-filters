# HSV Color Segmentation Converts the image to HSV space and isolates pixels within color ranges, allowing robust color-based masking and detection.

import cv2
import numpy as np

def hsv_segment(
    image,
    lower=np.array([0, 120, 70]),
    upper=np.array([10, 255, 255])
):
    """
    HSV color segmentation.
    Input: BGR image
    Output: BGR image with masked region
    """
    if image is None:
        raise ValueError("Input image is None")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result
