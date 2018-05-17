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
import dlib
import cv2
from imutils import face_utils

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


def detect_parts(image, filename):
	distances = []
	# resize the image, and convert it to grayscale
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale image
	rects = detector(gray, 1)
	
	# loop over the face detections
	for (i, rect) in enumerate(rects):
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		distances = euclidean_all(shape)
		#output = face_utils.visualize_facial_landmarks(image, shape)
		# visualize all facial landmarks with a transparent overlay
		#cv2.imshow("Image", output)
		#cv2.waitKey(0)
	return distances

def euclidean(a, b):
    dist = np.linalg.norm(a-b)
    return dist 

def euclidean_all(a):  # calculates distances between all 68 elements
	distances = ""
	for i in range(0, len(a)):
		for j in range(0, len(a)):
			dist = euclidean(a[i], a[j])
			dist = "%.2f" % dist;
			distances = distances + " " + str(dist)
	return distances

file_exists = os.path.isfile('testing.csv')
if(file_exists == True):
	os.remove('testing.csv')

for path in glob.glob('../data/kaggle/PublicTest/*/*.png'): # assuming png
	image = cv2.imread(path)
	path, filename = os.path.split(path)
	path, label = os.path.split(path)
	
	distances = detect_parts(image, filename)
	if distances != []:
		distances = np.asarray(distances)
		
		with open('testing.csv', 'ab') as csvfile:
    			fieldnames = ['emotion', 'vectors']
    			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
			if not file_exists:	
    				writer.writeheader()
   			writer.writerow({'emotion': int(label), 'vectors': distances})
			print label, filename




