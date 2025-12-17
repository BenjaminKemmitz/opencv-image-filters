# Laplacian Filter Uses the second derivative of pixel intensity to detect regions of rapid intensity change—useful for highlighting fine edges.

import cv2
import numpy as np

def laplacian_edge(image):
    lap = cv2.Laplacian(image, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))
    return lap
