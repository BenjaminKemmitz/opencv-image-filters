# Harris Corner Detector Detects corners by analyzing variations in intensity in local windows—useful for tracking and alignment tasks.

import cv2
import numpy as np

def harris_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    result = image.copy()
    result[corners > 0.01 * corners.max()] = [0, 0, 255]

    return result
