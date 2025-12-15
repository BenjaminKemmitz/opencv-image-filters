import cv2

def otsu_thresholding(image):
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

if __name__ == "__main__":
    result = otsu_threshold("../images/input/sample.jpg")
    cv2.imwrite("../images/output/otsu_threshold.jpg", result)

