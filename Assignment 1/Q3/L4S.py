import cv2
import numpy as np
import matplotlib.pyplot as plt 

img = cv2.imread('Cat_sleeping.jpg')

ksize1 = 5
ksize2 = 3
fimgS = cv2.medianBlur(img, ksize1)
fimgS = cv2.medianBlur(fimgS, ksize2)

cv2.imwrite('aL4S.jpg',fimgS)

plt.imshow(fimgS)
plt.show()