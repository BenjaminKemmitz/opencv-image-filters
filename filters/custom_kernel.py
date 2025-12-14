import cv2
import numpy as np

# Define filter function
def custom_kernel(image_path):
    img = cv2.imread(image_path)

    # Sharpen filter kernel
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    
    filtered = cv2.filter2D(img, -1, kernel)
    return filtered

# Check if main file
if __name__ == "__main__":
    path = "../images/input/sample.jpg"
    output_path = "../images/output/custom_sharpen.jpg"
  
    #Output sharper image through custom_filter function
    result = custom_filter(path)
    cv2.imwrite(output_path, result)
    print(f"Saved: {output_path}")
