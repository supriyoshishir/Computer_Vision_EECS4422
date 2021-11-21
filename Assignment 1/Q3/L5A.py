import cv2
import numpy as np
import matplotlib.pyplot as plt 

img = cv2.imread('Cat_sleeping.jpg')

ksize1 = 8
sigma1 = 2
ksize2 = 10
sigma2 = 3

gkern1 = cv2.getGaussianKernel(ksize1, sigma1)
gkern2 = cv2.getGaussianKernel(ksize2, sigma2)
fimgA = cv2.filter2D(img, -1, gkern1)
fimgA = cv2.filter2D(fimgA, -1, gkern2)

cv2.imwrite('aL5A.jpg',fimgA)

plt.imshow(fimgA)
plt.show()