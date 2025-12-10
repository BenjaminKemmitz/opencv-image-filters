from filters.adaptive_threshold import adaptive_threshold
import matplotlib.pyplot as plt
import cv2

img = adaptive_threshold("../images/input/sample.jpg")

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Adaptive Threshold Demo")
plt.axis("off")
plt.show()
