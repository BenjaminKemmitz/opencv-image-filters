# Basic Thresholding Converts an image to binary by comparing pixel intensity to a fixed threshold value.

import cv2

def basic_threshold(image, thresh=128):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
