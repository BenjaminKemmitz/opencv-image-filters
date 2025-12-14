import cv2

def gaussian_blur(image, ksize=(21, 21), sigma=0):
    return cv2.GaussianBlur(image, ksize, sigma)


# Optional standalone test
if __name__ == "__main__":
    img = cv2.imread("../images/input/sample.jpg")
    if img is None:
        raise ValueError("Could not load image")

    result = gaussian_blur(img)
    cv2.imwrite("../images/output/gaussian_blur.jpg", result)
    print("Saved: ../images/output/gaussian_blur.jpg")
