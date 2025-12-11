from filters.harris_corners import harris_corners
import matplotlib.pyplot as plt
import cv2

img = harris_corners("../images/input/sample.jpg")

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Harris Corners Demo")
plt.axis("off")
plt.show()
