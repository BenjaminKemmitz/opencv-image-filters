import cv2

def canny_edge(image, low=100, high=200):
    if image is None:
        raise ValueError("Input image is None")
        
    return cv2.Canny(image, low, high)

if __name__ == "__main__":
    edges = canny_edge("../images/input/sample.jpg")
    cv2.imwrite("../images/output/canny_edge.jpg", edges)



