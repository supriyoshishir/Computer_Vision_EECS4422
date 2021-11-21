import cv2
import numpy as np
import matplotlib.pyplot as plt 

plt.rcParams['figure.figsize'] = [16, 10]

image = cv2.imread('sample_letters.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel = np.ones((5, 5), np.uint8)
image_erosion = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel) 

image_new = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
cv2.imshow('image_x', image_new)
cv2.waitKey(0)

#plt.imshow(image_erosion, cmap='gray')
#plt.show()