from typing_extensions import Required
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from plutoainetwork.preprocessing import ImageToArrayPreprocessor
from plutoainetwork.preprocessing import SimplePreprocessor
from plutoainetwork.datasets import SimpleDatasetsLoader
from plutoainetwork.nn.conv import ShallowNet
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np 
import argparse

ap =  argparse.ArgumentParser()
ap.add("-d", "--dataset", required=True, help="path to image dataset")
args = vars(ap.parse_args())

print("Loading Images")
imagePaths = list(paths.list_images(args['dataset'])) 

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetsLoader(preprocessors=[sp, iap])

(data, labels) = sdl.load(imagePaths, verbose=500)
data =  data.astype('float')/255
