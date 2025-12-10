import cv2
import numpy as np

def morphology(image_path):
    img = cv2.imread(image_path, 0)
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5,5), np.uint8)

    eroded = cv2.erode(binary, kernel, iterations=1)
    dilated = cv2.dilate(binary, kernel, iterations=1)

    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return eroded, dilated, opening, closing

if __name__ == "__main__":
    e, d, o, c = morphology("../images/input/sample.jpg")
    cv2.imwrite("../images/output/eroded.jpg", e)
    cv2.imwrite("../images/output/dilated.jpg", d)
    cv2.imwrite("../images/output/opening.jpg", o)
    cv2.imwrite("../images/output/closing.jpg", c)
