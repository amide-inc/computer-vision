import numpy as np
import cv2
import os

class SimpleDatasetsLoader:

	def __init__(self, preprocessors=None):
		self.preprocessors = preprocessors

		if self.preprocessors is None:
			self.preprocessors = []

	def load(self, imagePaths, verbose=-1):
		data = []
		labels = []

		for (i, imagePath) in enumerate(imagePaths):
			image = cv2.imread(imagePath)
			label = imagePath.split(os.path.sep)[-2]

			if self.preprocessors is not None:
				for p in preprocessors:
					image = p.preprocess(image)

			data.append(image)
			labels.append(label)