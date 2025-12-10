import cv2
import numpy as np

def hsv_segment(image_path, lower, upper):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    return mask, result

if __name__ == "__main__":
    # Example: Detect red
    lower = np.array([0, 120, 70])
    upper = np.array([10, 255, 255])

    mask, seg = hsv_segment("../images/input/sample.jpg", lower, upper)
    cv2.imwrite("../images/output/hsv_mask.jpg", mask)
    cv2.imwrite("../images/output/hsv_segment.jpg", seg)

