#import guassian filter program
from filters.gaussian_blur import gaussian_blur
import matplotlib.pyplot as plt
import cv2

img = gaussian_blur("../images/input/sample.heic")

#pyplot data & display
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Gaussian Blur Demo")
plt.axis("off")
plt.show()
