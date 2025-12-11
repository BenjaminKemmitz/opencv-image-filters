from filters.morphology import morphology
import matplotlib.pyplot as plt
import cv2

img = morphology("../images/input/sample.jpg")

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Morphology Demo")
plt.axis("off")
plt.show()
