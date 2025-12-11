from filters.sobel_edge import sobel_edge
import matplotlib.pyplot as plt
import cv2

img = sobel_edge("../images/input/sample.jpg")

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Sobel Edge Demo")
plt.axis("off")
plt.show()
