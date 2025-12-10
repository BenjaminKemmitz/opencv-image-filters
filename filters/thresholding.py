import cv2

def basic_threshold(image_path, thresh=128):
    img = cv2.imread(image_path, 0)
    _, binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    return binary

if __name__ == "__main__":
    result = basic_threshold("../images/input/sample.jpg")
    cv2.imwrite("../images/output/basic_threshold.jpg", result)

