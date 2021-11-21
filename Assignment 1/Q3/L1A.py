import cv2
import numpy as np
import matplotlib.pyplot as plt 

img = cv2.imread('Cat_sleeping.jpg')

ksize1 = 10
sigma = 3
ksize2 = 5

gkern = cv2.getGaussianKernel(ksize1, sigma)
fimgA = cv2.filter2D(img, -1, gkern)
fimgA = cv2.medianBlur(fimgA, ksize2)

cv2.imwrite('aL1A.jpg',fimgA)

plt.imshow(fimgA)
plt.show()