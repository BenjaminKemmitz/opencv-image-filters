import cv2

def shi_tomasi(image, max_corners=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, max_corners, 0.01, 10)
    corners = corners.astype(int)

    for x, y in corners.reshape(-1, 2):
        cv2.circle(image, (x, y), 4, (0, 255, 0), -1)

    return image

if __name__ == "__main__":
    img = shi_tomasi("../images/input/sample.jpg")
    cv2.imwrite("../images/output/shi_tomasi.jpg", img)

