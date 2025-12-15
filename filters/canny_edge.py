import cv2

def canny_edge(image, low=100, high=200):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    binary = cv2.Canny(gray, image, low, high)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BRGR)

if __name__ == "__main__":
    edges = canny_edge("../images/input/sample.jpg")
    cv2.imwrite("../images/output/canny_edge.jpg", edges)



