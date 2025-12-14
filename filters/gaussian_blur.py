import cv2

# Reads input image, defines guassian blur
def gaussian_blur(image, ksize=(5, 5), sigma=0):
    blurred = cv2.GaussianBlur(image, ksize, sigma)
    return blurred

#Apply Gaussian Blur
if __name__ == "__main__":
    path = "../images/input/sample.jpg"
    output_path = "../images/output/gaussian_blur.jpg"
  
    result = gaussian_blur(path)
    cv2.imwrite(output_path, result)
    print(f"Saved: {output_path}")
