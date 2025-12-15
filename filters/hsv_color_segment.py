import cv2
import numpy as np

def hsv_segment(image, lower = np.array([0, 120, 70]), upper = np.array([10, 255, 255])):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)

    return mask, result

if __name__ == "__main__":
    # Example: Detect red
    lower = np.array([0, 120, 70])
    upper = np.array([10, 255, 255])

    mask, seg = hsv_segment("../images/input/sample.jpg", lower, upper)
    cv2.imwrite("../images/output/hsv_mask.jpg", mask)
    cv2.imwrite("../images/output/hsv_segment.jpg", seg)

