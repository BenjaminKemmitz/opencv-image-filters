from filters.bilateral_filter import bilateral_filter
import matplotlib.pyplot as plt
import cv2

img = gaussian_blur("../images/input/sample.jpg")

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Gaussian Blur Demo")
plt.axis("off")
plt.show()
