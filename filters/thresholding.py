import cv2

def basic_threshold(image_path, thresh=128):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

if __name__ == "__main__":
    result = basic_threshold("../images/input/sample.jpg")
    cv2.imwrite("../images/output/basic_threshold.jpg", result)

