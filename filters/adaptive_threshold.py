import cv2

def adaptive_threshold(image):
     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    # Convert back to BGR for pipeline consistency
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

if __name__ == "__main__":
    result = adaptive_threshold("../images/input/sample.jpg")
    cv2.imwrite("../images/output/adaptive_threshold.jpg", result)
