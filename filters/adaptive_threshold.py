import cv2

def adaptive_threshold(image_path):
    img = cv2.imread(image_path, 0)
    binary = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    return binary

if __name__ == "__main__":
    result = adaptive_threshold("../images/input/sample.jpg")
    cv2.imwrite("../images/output/adaptive_threshold.jpg", result)
