from filters.shi_tomasi import shi_tomasi
import matplotlib.pyplot as plt
import cv2

img = shi_tomasi("../images/input/sample.jpg")

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Shi-Tomasi Demo")
plt.axis("off")
plt.show()
