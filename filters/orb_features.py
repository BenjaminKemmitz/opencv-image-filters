#ORB Feature Detector A fast, rotation-invariant keypoint detector and descriptor generator used in SLAM, mapping, and object recognition.

import cv2

def orb_features(image, n_features=500):
    orb = cv2.ORB_create(nfeatures=n_features)

    keypoints, descriptors = orb.detectAndCompute(image, None)
    out = cv2.drawKeypoints(image, keypoints, None, color=(0,255,0))

    return out
