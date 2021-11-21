import cv2
import numpy as np
import matplotlib.pyplot as plt 

plt.rcParams['figure.figsize'] = [16, 10]

def img_out = Question_5(scene_img, pokemon_string, location, width)
img_pikachu = cv2.imread('Q5_train/Pikachu.jpg')
img_pikachu = cv2.cvtColor(img_pikachu, cv2.COLOR_BGR2RGB)

(y,x,c) = img_pikachu.shape

thresh = 150

mask = np.zeros((y,x)) 
mask = img_pikachu[:,:,1] < 1.5*img_pikachu[:,:,0]
mask = np.logical_or(mask, img_pikachu[:,:,1] < 1.5*img_pikachu[:,:,2])
mask = np.logical_or(mask, img_pikachu[:,:,1] < thresh)
plt.imshow(mask, cmap='gray')
plt.show()

#background1 = cv2.imread('Q5_samples/image_1297.png')
#background1 = cv2.cvtColor(background1, cv2.COLOR_BGR2RGB)
#background1 = cv2.resize(background1, (x,y))

# This line converts our alpha mask from a single channel image to three channels of identical values;
# this allows us to compute our image summation more easily as a matrix operation
#mask3c = cv2.cvtColor(mask.astype('uint8'), cv2.COLOR_GRAY2RGB)

#composite1 = img_pikachu*mask3c + background1*(1-mask3c)

#plt.imshow(composite1)
#plt.show()