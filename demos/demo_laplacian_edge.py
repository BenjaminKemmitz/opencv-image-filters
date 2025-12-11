from filters.laplacian_edge import laplacian_edge
import matplotlib.pyplot as plt
import cv2

img = laplacian_edge("../images/input/sample.jpg")

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Laplacian Edge Demo")
plt.axis("off")
plt.show()
