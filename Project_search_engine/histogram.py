import numpy as np
import cv2

class RGB_Histogram:
	def __init__(self, bins):
		# storing the number of bins in the histogram
		self.bins = bins

	def describe(self, image):
		# computes a 3D histogram in the RGB colorspace,
		histogram = cv2.calcHist([image], [0, 1, 2], None, self.bins, [0, 256, 0, 256, 0, 256])
		# normalizing the histogram so that images with the same content will have the same histogram
		# and returns the 3D histogram as a flattened array
		histogram = cv2.normalize(histogram, histogram).flatten()
		return histogram