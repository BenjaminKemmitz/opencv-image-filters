# Sobel Operator Computes gradient magnitude in the x and y directions to highlight vertical and horizontal edges.

import cv2
import numpy as np

# Function for image filter
def sobel_edge(image):

    # Compute x & y gradient directions
    # Arguments: src, ddepth, dx, dy, ksize
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude G= srt(Gx^2 +Gy^2)
    magnitude = cv2.magnitude(sobelx, sobely)
    #min-max normalization, scales floating point mag to 0-255 range for np.unit8 image
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))

    return magnitude
