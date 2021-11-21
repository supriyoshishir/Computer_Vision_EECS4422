import cv2
import numpy as np
import math

def Question_4_a(img_in, thick, border_colour, font_colour):
    image = img_in
    height, width, channels = image.shape

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find Canny edges
    edged = cv2.Canny(gray, 30, 200)
    indices = np.where(edged != [0])
    # make Canvas
    blank_image = np.zeros((height,width,3), np.uint8)

    # Put Arguments
    color = border_colour
    radius = 1
    thickness = thick

    #Draw Edge Pixels
    for i in range(len(indices[0])):
        cv2.circle(blank_image, (indices[1][i],indices[0][i]), radius, color, thickness)
    #Invert image
    blank_image[np.where((blank_image == [0,0,0]).all(axis = 2))] = [255,255,255]

    #Get Text Font
    text_font = cv2.bitwise_and(image,blank_image)

    #Colorize Font
    blank_image[np.where((text_font == [0,0,0]).all(axis = 2))] = list(font_colour)

    return blank_image

def Question_4_b(img_in, shadow_size, shadow_magnitude, orientation):
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

if input("") = Question_4_a
    img_out = Question_4_a(img_in, thick, border_colour, font_colour)
elif input("") = Question_4_b
    img_out = Question_4_b(img_in, shadow_size, shadow_magnitude, orientation)

img_in = cv2.imread('input.png')
thick = 1
border_colour = [0,0,200]
font_colour = [200,0,0]
shadow_size = 3
shadow_magnitude = 5
orientation = 135

cv2.imshow('Contours', img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()