import cv2
import numpy as np
import matplotlib.pyplot as plt 
%matplotlib inline

plt.rcParams['figure.figsize'] = [16, 10]

background = cv2.imread('scene1.png')
background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
(b,a,c) = background.shape
print (background.shape)

image = cv2.imread('Flygon.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

(y,x,z) = image.shape
print (image.shape)

thresh = 50
mask = np.zeros((y,x))
mask = image[:,:,1] < 1.2*image[:,:,0]
mask = np.logical_or(mask, image[:,:,1] < 1.7*image[:,:,2])
mask = np.logical_or(mask, image[:,:,1] < thresh)

morph_size = 10
morph_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size,morph_size))
mask = cv2.erode(mask.astype(np.uint8), morph_kern, iterations=1)
mask = cv2.dilate(mask, morph_kern, iterations=1)
plt.imshow(mask, cmap='gray')

mask_new = cv2.cvtColor(mask.astype('uint8'), cv2.COLOR_GRAY2RGB)
composite = image*mask_new + background*(1-mask_new)
plt.imshow(composite)