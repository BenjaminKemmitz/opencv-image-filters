import cv2

def shi_tomasi(image, max_corners=100):
    """
    Shi-Tomasi corner detection (visual overlay).
    Input: BGR image
    Output: BGR image with corners drawn
    """
    if image is None:
        raise ValueError("Input image is None")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, max_corners, 0.01, 10)

    # IMPORTANT: work on a copy
    result = image.copy()

    if corners is not None:
        corners = corners.astype(int)
        for x, y in corners.reshape(-1, 2):
            cv2.circle(result, (x, y), 4, (0, 255, 0), -1)

    return result

if __name__ == "__main__":
    img = shi_tomasi("../images/input/sample.jpg")
    cv2.imwrite("../images/output/shi_tomasi.jpg", img)

