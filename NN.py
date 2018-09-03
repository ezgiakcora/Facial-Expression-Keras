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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

num_classes = 7 #'Angry','Disgust', 'Fear', 'Happy', 'Neutral', Sad', 'Surprise'
batch_size = 64
epochs = 300
fit = True
seed(8)

# --------------------------------------------------------------------
# Data preperation from csv file
def prepare_data(fileName):
	X, Y = [], []
	x_train, y_train, x_test, y_test = [], [], [], []
	with open(fileName) as f:
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

			X.append(val)
			Y.append(emotion)
				
		x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=4)

		print ("********Training set size: ", str(len(x_train)))

		x_train = np.array(x_train)
		y_train = np.array(y_train)
		
		
		minmax = preprocessing.MinMaxScaler()
		x_train = minmax.fit_transform(x_train)
		
		x_test = np.array(x_test)
		y_test = np.array(y_test)

		# Normalization of the testing data   
		minmax = preprocessing.MinMaxScaler()
		x_test = minmax.fit_transform(x_test)

		return x_train, y_train, x_test, y_test

# --------------------------------------------------------------------
# Training and test data preperation  
x_train, y_train, x_test, y_test = prepare_data("./data/JAFFE_CK.csv")

print ("Training set size: ", str(len(x_train)))
print ("Test set size: ", str(len(x_test)))
# --------------------------------------------------------------------
# Construct the NN structure
model = Sequential()
#1st layer
model.add(Dense(512, input_shape=(4624,)))
model.add(Dense(512))
model.add(Dense(512))
model.add(Dense(512))
model.add(Dense(512))
model.add(Dense(num_classes, activation='softmax'))

# compile the model
opt = SGD(lr=0.0001, decay=1e-6, momentum=0.9)
# opt = keras.optimizers.Adam(lr=0.00001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model.fit(x_train, # Features
                      y_train, # Target
                      epochs=epochs, # Number of epochs
                      verbose=1, # No output
                      batch_size=batch_size, # Number of observations per batch
                      validation_data=(x_test, y_test)) # Data for evaluation

# --------------------------------------------------------------------
# Visualize the training and test loss through epochs

# Get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();

y_pred = model.predict_classes(x_test)
y_true = [0] * len(y_pred)

for i in range(0, len(y_test)):
	max_index = np.argmax(y_test[i])
	y_true[i] = max_index

# --------------------------------------------------------------------
# Print wrong classifications 

#for i in range(len(y_pred)):
#	if(y_pred[i] != y_true[i]):
#		print(str(i) + ' --> Predicted: ' +  str(y_pred[i]) + " Expected: " + str(y_true[i]))

# --------------------------------------------------------------------
# Draw the confusion matrix 
cm = confusion_matrix(y_pred, y_true, labels=range(num_classes))

labels = ['Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# --------------------------------------------------------------------
# Evaluate the model on the test set
scores = model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# --------------------------------------------------------------------
# Save the model and the weights 
#model_json = model.to_json()
#with open("./model/model.json", "w") as json_file:
#    json_file.write(model_json)
#model.save_weights("./model/model.h5")
#print("Saved model to disk")
