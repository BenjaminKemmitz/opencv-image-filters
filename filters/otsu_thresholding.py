# Otsu’s Thresholding Automatically computes the optimal threshold by maximizing inter-class variance—ideal when the image histogram is bimodal.

import cv2

def otsu_thresholding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
