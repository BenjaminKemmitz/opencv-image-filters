from filters.canny_edge import canny_edge
import matplotlib.pyplot as plt
import cv2

img = canny_edge("../images/input/sample.jpg")

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Canny Edge Demo")
plt.axis("off")
plt.show()
