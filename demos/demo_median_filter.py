from filters.median_filter import median_filter
import matplotlib.pyplot as plt
import cv2

img = median_filter("../images/input/sample.jpg")

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Median Filter Demo")
plt.axis("off")
plt.show()
