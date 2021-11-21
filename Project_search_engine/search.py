# python search.py --dataset images --index index.pickle --query query/qi1.jpg

from histogram import RGB_Histogram
from metric import Searcher
import numpy as np
import argparse
import pickle
import cv2
# from imutils import build_montages
# from imutils import paths
import time
startTime = time.time()

# constructing the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required = True, help = "path to the images directory")
parser.add_argument("-i", "--index", required = True, help = "path to the stored index of the images")
parser.add_argument("-q", "--query", required = True, help = "path to the query image")
args = vars(parser.parse_args())

# loading the query image and display it
queryImage = cv2.imread(args["query"])
cv2.imshow("Query", queryImage)
print ("query: %s" % (args["query"]))

# describing the query in the image with 8 bins per channel
image_descrption = RGB_Histogram([8, 8, 8])
queryFeatures = image_descrption.describe(queryImage)

# loading the index to perform the search
location = open(args["index"], "rb").read()
index = pickle.loads(location, encoding = "latin1")
results = Searcher(index).search(queryFeatures)

# creating a window to display the top 4 results from a dataset of 16 indexed images
result_window = np.zeros((166 * 4, 400, 3), dtype = "uint8")

# loop over the top 4 results
for j in range(0, 4):
	# get the result using row-major order and load the resulting image
	(score, imageName) = results[j]
	path = args["dataset"] + "/%s" % (imageName)
	result = cv2.imread(path)
	print ("%d. %s : %.3f" % (j+1, imageName, score))

	if j < 4:
		result_window[j*166 : (j+1)*166, : ] = result
 
# displaying the results
cv2.imshow("Results", result_window)
cv2.waitKey(0)
cv2.destroyAllWindows()

endTime = time.time()
print ("Execution Time:", (endTime - startTime))