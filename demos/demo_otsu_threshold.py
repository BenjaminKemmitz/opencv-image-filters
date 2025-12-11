from filters.otsu_threshold import otsu_threshold
import matplotlib.pyplot as plt
import cv2

img = otsu_theshold("../images/input/sample.jpg")

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Otsu Threshold Demo")
plt.axis("off")
plt.show()
