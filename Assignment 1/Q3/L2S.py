import cv2
import numpy as np
import matplotlib.pyplot as plt 

img = cv2.imread('Cat_sleeping.jpg')

ksize1 = 5
sigma = 3
ksize2 = 10

fimgS = cv2.medianBlur(img, ksize1)
gkern = cv2.getGaussianKernel(ksize2, sigma)
fimgS = cv2.filter2D(fimgS, -1, gkern)

cv2.imwrite('aL2S.jpg',fimgS)

plt.imshow(fimgS)
plt.show()