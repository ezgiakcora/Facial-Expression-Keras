# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
from PIL import Image
import dlib
import cv2
import glob
import scipy.misc
import os
import csv 
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
import scipy.misc
from sklearn import preprocessing
import dlib
import cv2
from imutils import face_utils
import math
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image


# initialize dlib's face detector and create a predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# calculate distance vectors between each point and the center

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

def detect_parts(image, filename):
	distances = []
	# resize the image, and convert it to grayscale
	image = imutils.resize(image, width=200, height=200)
	
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale image
	rects = detector(gray, 1)
	
	# loop over the face detections
	for (i, rect) in enumerate(rects):
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		distances = euclidean_all(shape)
		# output = face_utils.visualize_facial_landmarks(image, shape)
		# visualize all facial landmarks with a transparent overlay
		# cv2.imshow("Image", output)
		# cv2.waitKey(0)	
	return distances

def euclidean(a, b):
    dist = math.sqrt(math.pow((b[0] - a[0]), 2) + math.pow((b[1] - a[1]), 2))
    return dist 

def euclidean_all(a):  # calculates distances between all 68 elements
	distances = ""
	for i in range(0, len(a)):
		for j in range(0, len(a)):
			dist = euclidean(a[i], a[j])
			dist = "%.2f" % dist;
			distances = distances + " " + str(dist)
	return distances



def generate_csv(dirName, csvName):
	file_exists = os.path.isfile(csvName)
	if(file_exists == True):
		os.remove(csvName)

	for path in glob.glob(dirName): # assuming png
		img = cv2.imread(path)
		path, filename = os.path.split(path)
		path, label = os.path.split(path)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		# print(faces) #locations of detected faces

		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
			detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
			distances = detect_parts(detected_face, filename)
			if distances != []:
				distances = np.asarray(distances)
			
				with open(csvName, 'a', encoding='utf-8-sig') as csvfile:
					fieldnames = ['emotion', 'vectors']
					writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
					file_exists = os.path.isfile(csvName)
					if not file_exists:	
						writer.writeheader()
					writer.writerow({'emotion': label, 'vectors': distances})
					print (label, filename)
			
			cv2.imshow('img',img)
			cv2.waitKey(1)
		

# ---------------------------------------------------------------------------------------------------


testing_csvName = "JAFFE_CK.csv"
testing_dirName = '../data/CK_JAFFE/*/*'
generate_csv(testing_dirName, testing_csvName)

print("Success: CSV files are generated!")

# ---------------------------------------------------------------------------------------------------








