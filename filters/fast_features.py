# FAST Corner Detector Extremely fast corner detector ideal for real-time vision tasks, especially in embedded or robotics systems.

import cv2

def fast_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(gray, None)

    out = cv2.drawKeypoints(image, keypoints, None, color=(255,0,0))
    return out
