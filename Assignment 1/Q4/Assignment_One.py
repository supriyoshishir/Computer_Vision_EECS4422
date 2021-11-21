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

mask_thres={"Flygon":2.16,
            "Jigglypuff":1.7,
            "Muk":1.7,
            "Pikachu":1.7
            }

def create_mask(img,pokemon_string):
    image = img
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    (y,x,z) = image.shape

    thresh = 50
    mask = np.zeros((y,x))
    mask = image[:,:,1] < 1.2*image[:,:,0]
    mask = np.logical_or(mask, image[:,:,1] < mask_thres[pokemon_string]*image[:,:,2])
    mask = np.logical_or(mask, image[:,:,1] < thresh)

    morph_size = 10
    morph_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size,morph_size))
    mask = cv2.erode(mask.astype(np.uint8), morph_kern, iterations=1)
    mask = cv2.dilate(mask, morph_kern, iterations=1)
    return mask

def image_resize(image, width):
    # image size
    (h, w) = image.shape[:2]
    # calculate aspect ratio
    r = width / float(w)
    dim = (width, int(h * r))
    # resize the image
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    # return the resized image
    return resized

def Question_5(scene_img, pokemon_string, location, width):
    #Get foreground and background
    background = cv2.imread(scene_img)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    (b,a,c) = background.shape
    pokemon = cv2.imread(pokemon_string+".jpg")
    image = pokemon

    # Get location x,y
    loc_x=location[0]
    loc_y=location[1]

    #Create mask of Pikachu image
    mask = create_mask(image,pokemon_string)

    #Find largest blob/contour... Segment Pikachu out
    contours, hierarchy = cv2.findContours(mask,
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        #find the biggest area of the contour
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        crop_img = image[y:y+h, x:x+w]

    #Resize to required dimension
    crop_img = image_resize(crop_img, width)
    foreground=crop_img

    # Create Alpha Channels (Positive and Negative)
    cropped_mask=create_mask(crop_img,pokemon_string)
    cropped_mask[cropped_mask > 0] = 255
    alpha=cropped_mask
    alpha_inv=255-alpha

    #Create region of interest
    rows,cols,channels = foreground.shape
    roi = background[loc_y:rows+loc_y, loc_x:cols+loc_x ]

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = alpha_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(foreground,foreground,mask = alpha)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    background[loc_y:rows+loc_y, loc_x:cols+loc_x ] = dst

    return background

question_input = input("")

if question_input == "Question_4_a":
    img_in = cv2.imread('input.png')
    thick = int(input())
    border_colour = list(map(int,input().split()))
    font_colour = list(map(int,input().split()))
    img_out = Question_4_a(img_in, thick, border_colour, font_colour)

elif question_input == "Question_4_b":
    img_in = cv2.imread('input.png')
    shadow_size = int(input())
    shadow_magnitude = int(input())
    orientation = int(input())
    img_out = Question_4_b(img_in, shadow_size, shadow_magnitude, orientation)
    
elif question_input == "Question_5": 
    scene_img = input()
    pokemon_string = input()
    location = list(map(int,input().split()))
    width = int(input())
    img_out = Question_5(scene_img, pokemon_string, location, width)

cv2.imshow('Contours', img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()