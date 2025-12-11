from filters.custom_kernel import custom_kernel
import matplotlib.pyplot as plt
import cv2

img = custom_kernel("../images/input/sample.jpg")

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Custom Kernel Demo")
plt.axis("off")
plt.show()
