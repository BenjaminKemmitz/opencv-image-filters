# Median Filter Replaces each pixel with the median of its neighborhood, highly effective for removing salt-and-pepper noise.

import cv2

# Define median filter function, run through medianBlur operator
def median(image, ksize=5):
    if image is None:
        raise ValueError("Input image is None")
    
    filtered = cv2.medianBlur(image, ksize)
    return filtered
