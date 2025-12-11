from filters.hsv_segmentation import hsv_segmentation
import matplotlib.pyplot as plt
import cv2

img = hsv_segmentation("../images/input/sample.jpg")

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("HSV Segmentation Demo")
plt.axis("off")
plt.show()
