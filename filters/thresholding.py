import cv2

def basic_threshold(image_path, thresh=128):
    _, binary = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    return binary

if __name__ == "__main__":
    result = basic_threshold("../images/input/sample.jpg")
    cv2.imwrite("../images/output/basic_threshold.jpg", result)

