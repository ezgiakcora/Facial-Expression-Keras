import numpy as np
import cv2
from keras.preprocessing import image
import dlib
from imutils import face_utils
import imutils
from sklearn import preprocessing
import math
from keras.models import model_from_json
#-----------------------------
#opencv initialization
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

#-----------------------------
#face expression recognizer initialization
# Using pretrained model
model = model_from_json(open("./model/model.json", "r").read())
model.load_weights('./model/model.h5') #load weights

#-----------------------------

emotions = ( 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise')
# initialize dlib's face detector and create a predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def detect_parts(image):
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
		# visualize all facial landmarks with a transparent overlay
		#output = face_utils.visualize_facial_landmarks(image, shape)
		#cv2.imshow("Image", output)
		#cv2.waitKey(0)	
	return distances

def euclidean(a, b):
    dist = math.sqrt(math.pow((b[0] - a[0]), 2) + math.pow((b[1] - a[1]), 2))
    return dist 

# calculates distances between all 68 elements
def euclidean_all(a):  
	distances = ""
	for i in range(0, len(a)):
		for j in range(0, len(a)):
			dist = euclidean(a[i], a[j])
			dist = "%.2f" % dist;
			distances = distances + " " + str(dist)
	return distances


while(True):
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
		detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
		distances = detect_parts(detected_face)

		if(len(distances)!=0):
			val = distances.split(" ")[1:]
			val = np.array(val)
			val = val.astype(np.float)
			val = np.expand_dims(val, axis = 1)			
			minmax = preprocessing.MinMaxScaler()
			val = minmax.fit_transform(val)
			val = val.reshape(1,4624)

			predictions = model.predict(val) #store probabilities of 6 expressions
		#find max indexed array ( 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise')
			print ("Angry: %", predictions[0][0]/1.0 * 100)
			print ("Disgust: %", predictions[0][1]/1.0 * 100)
			print ("Fear: %", predictions[0][2]/1.0 * 100)
			print ("Happy: %", predictions[0][3]/1.0 * 100)
			print ("Neutral: %", predictions[0][4]/1.0 * 100)
			print ("Sad: %", predictions[0][5]/1.0 * 100)	
			print ("Surprised: %", predictions[0][6]/1.0 * 100)		
			print ("----------------------"	)	
			max_index = np.argmax(predictions[0])
			emotion = emotions[max_index]
		
			#write emotion text above rectangle
			cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
		
	cv2.imshow('img',img)

	if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		break

#kill open cv things		
cap.release()
cv2.destroyAllWindows()
