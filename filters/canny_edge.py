import cv2

def canny_edge(image, low=100, high=200):
    """
    Apply Canny edge detection.
    Input: BGR image (H, W, 3)
    Output: BGR image (H, W, 3)
    """
    if image is None:
        raise ValueError("Input image is None")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny
    edges = cv2.Canny(gray, low, high)

    # Convert back to BGR for pipeline consistency
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


# Standalone test
if __name__ == "__main__":
    img = cv2.imread("../images/input/sample.jpg")
    if img is None:
        raise ValueError("Could not load input image")

    edges = canny_edge(img)
    cv2.imwrite("../images/output/canny_edge.jpg", edges)
    print("Saved: ../images/output/canny_edge.jpg")

