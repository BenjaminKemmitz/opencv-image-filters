import cv2

# Define median filter function, run through medianBlur operator
def median_filter(image_path, ksize=5):
    img = cv2.imread(image_path)
    filtered = cv2.medianBlur(img, ksize)
    return filtered

# Check if main, run function
if __name__ == "__main__":
    result = median_filter("../images/input/sample.jpg")
    cv2.imwrite("../images/output/median_filter.jpg", result)
