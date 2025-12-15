import cv2
import numpy as np

# Function for image filter
def sobel_edge(image):

    # Compute x & y gradient directions
    # Arguments: src, ddepth, dx, dy, ksize
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude G= srt(Gx^2 +Gy^2)
    magnitude = cv2.magnitude(sobelx, sobely)
    #min-max normalization, scales floating point mag to 0-255 range for np.unit8 image
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))

    return magnitude

# Check for main program or being imported as module
if __name__ == "__main__":
    path = "../images/input/sample.jpg"
    output_path = "../images/output/sobel_edge.jpg"
    
    edges = sobel_edge(path)
    cv2.imwrite(output_path, edges)
    print(f"Saved: {output_path}")

