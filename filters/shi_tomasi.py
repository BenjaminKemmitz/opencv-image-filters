import cv2

def shi_tomasi(image_path, max_corners=100):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, max_corners, 0.01, 10)
    corners = corners.astype(int)

    for x, y in corners.reshape(-1, 2):
        cv2.circle(img, (x, y), 4, (0, 255, 0), -1)

    return img

if __name__ == "__main__":
    img = shi_tomasi("../images/input/sample.jpg")
    cv2.imwrite("../images/output/shi_tomasi.jpg", img)

