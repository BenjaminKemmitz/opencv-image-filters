import cv2

# Define median filter function, run through medianBlur operator
def median(image, ksize=5):
    if image is None:
        raise ValueError("Input image is None")
    
    filtered = cv2.medianBlur(imgage, ksize)
    return filtered

# Optional standalone test
if __name__ == "__main__":
    img = cv2.imread("../images/input/sample.jpg")
    if img is None:
        raise ValueError("Could not load image")

    result = median(img)
    cv2.imwrite("../images/output/median_filter.jpg", result)
    print("Saved: ../images/output/median.jpg")
