import cv2
import numpy as np

def harris_corners(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    img[corners > 0.01 * corners.max()] = [0, 0, 255]

    return img

if __name__ == "__main__":
    img = harris_corners("../images/input/sample.jpg")
    cv2.imwrite("../images/output/harris_corners.jpg", img)

