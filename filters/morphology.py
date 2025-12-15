import cv2
import numpy as np

def morphology(image):
    """
    Morphological operations demo (opening + closing).
    Input: BGR image
    Output: BGR image
    """
    if image is None:
        raise ValueError("Input image is None")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to binary
    _, binary = cv2.threshold(
        gray, 128, 255, cv2.THRESH_BINARY
    )

    kernel = np.ones((5, 5), np.uint8)

    # Apply morphology
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Combine results side-by-side (visual clarity)
    combined = cv2.hconcat([opening, closing])

    # Convert back to BGR for pipeline consistency
    return cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)

if __name__ == "__main__":
    e, d, o, c = morphology("../images/input/sample.jpg")
    cv2.imwrite("../images/output/eroded.jpg", e)
    cv2.imwrite("../images/output/dilated.jpg", d)
    cv2.imwrite("../images/output/opening.jpg", o)
    cv2.imwrite("../images/output/closing.jpg", c)
