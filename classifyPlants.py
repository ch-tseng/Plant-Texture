#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from skimage import feature
from imutils import paths
import numpy as np
import argparse
import mahotas
import cv2

def describe(image, ftype="Hara"):
        #上面指令是從圖片的HSV色彩模型中，取得其平均值及標準差（有RGB三個channels，因此會各有3組平均值及標準差）作為特徵值
	(means, stds) = cv2.meanStdDev(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        #進行降維處理：將means及stds各3組array使用concatenate指令合成1組，再予以扁平化（變成一維）。
	colorStats = np.concatenate([means, stds]).flatten()

        #將圖片轉為灰階
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if(ftype=="Hara"):
            #取Haralick紋理特徵(texture features)的平均值
    	    haralick = mahotas.features.haralick(gray).mean(axis=0)

            #使用np.hstack將兩個一維的特徵陣列colorStats及haralick合成一個
    	    return np.hstack([colorStats, haralick])
        
	else:
            numPoints = 30
            radius = 3
            eps = 1e-7
            lbp = feature.local_binary_pattern(gray, numPoints, radius, method="uniform")
            (hist, _) = np.histogram(lbp.ravel(), bins=range(0, numPoints + 3), range=(0, numPoints + 2))

            # normalize the histogram
            hist = hist.astype("float")
            hist /= (hist.sum() + eps)

            return np.hstack([colorStats, hist])
        

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", required=True, help="LBA or Hara?")
ap.add_argument("-d", "--dataset", required=True, help="path to 8 scene category dataset")
ap.add_argument("-f", "--forest", type=int, default=-1,
	help="whether or not a Random Forest should be used")
args = vars(ap.parse_args())

print("[INFO] extracting features...")
imagePaths = sorted(paths.list_images(args["dataset"]))
labels = []
data = []

for imagePath in imagePaths:
        label = imagePath.split("/")[-2]
	image = cv2.imread(imagePath)

	features = describe(image)
	labels.append(label)
	data.append(features)

# construct the training and testing split by taking 75% of the data for training
# and 25% for testing
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data),
	np.array(labels), test_size=0.20, random_state=42)

model = RandomForestClassifier(n_estimators=args['forest'], random_state=42)

# train the decision tree
print("[INFO] training model...")
model.fit(trainData, trainLabels)

# evaluate the classifier
print("[INFO] evaluating...")
predictions = model.predict(testData)
print(classification_report(testLabels, predictions))

for i in np.random.randint(0, high=len(imagePaths), size=(10,)):
	imagePath = imagePaths[i]
	filename = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)
	features = describe(image)
	prediction = model.predict(features.reshape(1, -1))[0]

	# show the prediction
	print("[PREDICTION] {}: {}".format(filename, prediction))
	cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
	cv2.imshow("Image #"+str(i), image)

cv2.waitKey(0)
