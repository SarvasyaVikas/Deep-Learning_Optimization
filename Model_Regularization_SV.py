from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

def extract_color_histogram(image, bins=(8, 8, 8)):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])

	if imutils.is_cv2():
		hist = cv2.normalize(hist)

	else:
		cv2.normalize(hist, hist)

	return hist.flatten()

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True)
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["dataset"]))

sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose = 500)
data = data.reshape((data.shape[0], 3072))

le = LabelEncoder()
labels = le.fit_transform(labels)

print("[INFO] constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data), labels, test_size=0.25, random_state=42)

for r in (None, "l1", "l2"):
	print("[INFO] training model with `{}` penalty".format(r))
	model = SGDClassifier(loss="log", penalty=r, max_iter = 10, learning_rate = "constant", tol=1e-3, eta0 = 0.01, random_state = 12)
	model.fit(trainData, trainLabels)

	acc = model.score(testData, testLabels)
	print("[INFO] `{}` penalty accuracy: {:.2f}%".format(r, acc * 100))
