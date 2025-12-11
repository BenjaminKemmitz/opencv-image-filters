from filters.orb_features import orb_features
import matplotlib.pyplot as plt
import cv2

img = orb_features("../images/input/sample.jpg")

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Orb Features Demo")
plt.axis("off")
plt.show()
