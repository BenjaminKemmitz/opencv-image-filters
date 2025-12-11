from filters.thresholding import thresholding
import matplotlib.pyplot as plt
import cv2

img = thresholding("../images/input/sample.jpg")

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Thresholding Demo")
plt.axis("off")
plt.show()
