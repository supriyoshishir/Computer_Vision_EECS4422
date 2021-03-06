import cv2
import numpy as np

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

img_in = cv2.imread('input.png')
# Arguments
thick = 1
border_colour = [0,0,200]
font_colour = [200,0,0]

img_out = Question_4_a(img_in, thick, border_colour, font_colour)
cv2.imshow('Contours', img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()