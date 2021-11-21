# python index.py --dataset images --index index.pickle

from histogram import RGB_Histogram
import argparse
import cv2
import glob
import numpy as np
import pickle
import time

startTime = time.time()
# constructing the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required = True, help = "path to the indexed images directory")
parser.add_argument("-i", "--index", required = True, help = "path to the stored index of the images")
args = vars(parser.parse_args())
# initializing the index dictionary to store the quantifed images
index = {}
number = len(index)
# initializing the image descriptor by a 3D RGB space histogram with 8 bins per channel
image_descrption = RGB_Histogram([8, 8, 8])

# using glob to get the path of the images and loop over them
for imagePath in glob.glob(args["dataset"] + "/*jpg"):
	# extract our unique image ID (i.e. the filename)
	# imagePath = imagePath.replace("\\", "/") # for windows OS
	k = imagePath[imagePath.rfind("/") + 1:]
	# images are loaded and described using the RGB histogram descriptor
	image = cv2.imread(imagePath)
	features = image_descrption.describe(image)
	# updating the index
	index[k] = features

# writing the image features to a pickle file in binary mode
f = open(args["index"], "wb")
f.write(pickle.dumps(index))
f.close()

endTime = time.time()
print ("Execution Time:", (endTime - startTime))
print ("A pickle file has been created.")