# Custom Sharpening Filter Enhances edges and fine details using a manually designed convolution kernel to increase image crispness.

import cv2
import numpy as np

# Define filter function
def custom_kernel(image):
    # Sharpen filter kernel
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    
    filtered = cv2.filter2D(image, -1, kernel)
    return filtered
