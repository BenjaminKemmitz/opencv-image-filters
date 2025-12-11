from filters.fast_features import fast_features
import matplotlib.pyplot as plt
import cv2

img = fast_features("../images/input/sample.jpg")

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("FAST Features Demo")
plt.axis("off")
plt.show()
