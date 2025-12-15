import cv2
import numpy as np

def laplacian_edge(image):
    lap = cv2.Laplacian(image, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))
    return lap

if __name__ == "__main__":
    result = laplacian_edge("../images/input/sample.jpg")
    cv2.imwrite("../images/output/laplacian_edge.jpg", result)
