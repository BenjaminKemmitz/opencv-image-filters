import cv2

def bilateral_filter(image, d=9, sigmaColor=75, sigmaSpace=75):
    filtered = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
    return filtered

if __name__ == "__main__":
    result = bilateral_filter("../images/input/sample.jpg")
    cv2.imwrite("../images/output/bilateral_filter.jpg", result)
