import cv2
import numpy as np
import math

def  Question_4_b(img_in, shadow_size, shadow_magnitude, orientation):
    image = img_in
    height, width, channels = image.shape
    blank_image = np.zeros((height,width,1), np.uint8)

    num_rows, num_cols = image.shape[:2]

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find Canny edges
    edged = 255-cv2.Canny(gray, 30, 200)
    indices = np.where(edged != [0])

    radius=5
    x=radius*math.cos(orientation-90)
    y=radius*math.sin(orientation-90)

    translation_matrix = np.float32([ [1,0,x], [0,1,y] ])
    img_translation = cv2.warpAffine(edged, translation_matrix, (num_cols, num_rows), borderValue=(255,255,255))

    img = 255-cv2.subtract(gray,img_translation)

    kernel = np.ones((shadow_size,shadow_size), np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=1)
    blurImg = cv2.blur(img_erosion,(10-shadow_magnitude,10-shadow_magnitude))

    return blurImg

img_in = cv2.imread('input.png')
# Arguments
shadow_size = 3
shadow_magnitude = 5
orientation = 135

img_out= Question_4_b(img_in, shadow_size, shadow_magnitude, orientation)
cv2.imshow('Contours', img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()

