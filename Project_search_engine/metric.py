import numpy as np

class Searcher:
	def __init__(self, index):
		# storing the index of images
		self.index = index

	def search(self, queryFeatures):
		# initializing the dictionary of results
		results = {}

		# loop over the index
		for (k, features) in self.index.items():
			# d = self.chi2_distance(features, queryFeatures)
			d = self.euclidean_distance(features, queryFeatures)
			results[k] = d
		# sorting the results with more relevant images at the top of the list
		results = sorted([(v, k) for (k, v) in results.items()])
		# return the results
		return results

#	def chi2_distance(self, A, B, eps = 1e-10):
#		# compute the chi-squared distance
#		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
#			for (a, b) in zip(A, B)])
#		# return the chi-squared distance
#		return d

	def euclidean_distance(self, A, B):
		# compute the euclidean distance
		d = 0
		for (a, b) in zip(A, B):
			d += ((a - b) ** 2) ** 0.5
		# return the euclidean distance
		return d