from typing_extensions import Required
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras import optimizers
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


trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("compiling model")
opt = SGD(learning_rate=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("training network")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=1)
