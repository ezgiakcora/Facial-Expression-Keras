import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
import scipy.misc
import dlib
import cv2
from sklearn import preprocessing
from imutils import face_utils
import numpy as np
from keras.optimizers import SGD

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from numpy.random import seed

num_classes = 7 #angry, disgust, fear, happy, sad, surprise, neutral
batch_size = 64
epochs = 100 
fit = True
seed(8)

x_train, y_train, x_test, y_test = [], [], [], []

with open("./training_latest.csv") as f:
    	content = f.readlines()
	lines = np.array(content)
	num_of_instances = lines.size

for i in range(1,num_of_instances):   	
	emotion, distances = lines[i].split(",")
	distances = distances.strip() 
	val = distances.split(" ")
	val = np.array(val)
	val = val.astype(np.float)

	emotion = keras.utils.to_categorical(emotion, num_classes)
        y_train.append(emotion)
        x_train.append(val)


x_train = np.array(x_train)
y_train = np.array(y_train)

minmax = preprocessing.MinMaxScaler()
x_train = minmax.fit_transform(x_train)
#print x_train

with open("./testing_latest.csv") as f:
    	content = f.readlines()
	lines = np.array(content)
	num_of_instances = lines.size

for i in range(1,num_of_instances):   	
	emotion, distances = lines[i].split(",")
	distances = distances.strip() 
	val = distances.split(" ")
	val = np.array(val)
	val = val.astype(np.float)

	emotion = keras.utils.to_categorical(emotion, num_classes)
        y_test.append(emotion)
        x_test.append(val)


x_test = np.array(x_test)
y_test = np.array(y_test)
        
minmax = preprocessing.MinMaxScaler()
x_test = minmax.fit_transform(x_test)

#------------------------------

#construct NN structure
model = Sequential()

#1st layer
model.add(Dense(512, input_shape=(4624,)))
model.add(Dense(512))
model.add(Dense(512))
model.add(Dense(512))
model.add(Dense(512))
model.add(Dense(num_classes, activation='softmax'))

# compile the model
#opt = SGD(lr=0.0001, decay=1e-6, momentum=0.9)
opt = keras.optimizers.Adam(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#fit the model
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

#evaluate the model
scores = model.evaluate(x_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#overall evaluation
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', 100*score[1])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
