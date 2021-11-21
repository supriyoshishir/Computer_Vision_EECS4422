import cv2
import numpy as np
import matplotlib.pyplot as plt 

img = cv2.imread('Cat_sleeping.jpg')

ksize1 = 3
ksize2 = 5
fimgA = cv2.medianBlur(img, ksize1)
fimgA = cv2.medianBlur(fimgA, ksize2)

cv2.imwrite('aL3A.jpg',fimgA)

plt.imshow(fimgA)
plt.show()