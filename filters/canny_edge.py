import cv2

def canny_edge(image_path, low=100, high=200):
    img = cv2.imread(image_path, 0)
    edges = cv2.Canny(img, low, high)
    return edges

if __name__ == "__main__":
    edges = canny_edge("../images/input/sample.jpg")
    cv2.imwrite("../images/output/canny_edge.jpg", edges)
