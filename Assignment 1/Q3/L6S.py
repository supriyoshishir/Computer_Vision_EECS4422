import cv2
import numpy as np
import matplotlib.pyplot as plt 

img = cv2.imread('Cat_sleeping.jpg')

ksize1 = 8
sigma1 = 2
ksize2 = 10
sigma2 = 3

gkern = cv2.getGaussianKernel(ksize1, sigma1)
fimgS = cv2.filter2D(img, -1, gkern)
gkern = cv2.getGaussianKernel(ksize2, sigma2)
fimgS = cv2.filter2D(fimgS, -1, gkern)

cv2.imwrite('aL6S.jpg',fimgS)

plt.imshow(fimgS)
plt.show()