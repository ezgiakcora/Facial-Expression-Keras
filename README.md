# Facial-Expression-Keras
The aim of this project is to recognize facial expression from a video streaming by using deep learning.
It takes pictures or webcam video as input. It detects all faces in each frame, and then classifies the 
expressions to one of the 7 classes: Angry, Disgusted, Neutral, Sad, Happy, Surprised and Fear. 

The algorithm is based on extracting 68 facial landmarks with dlib and use the eucledian distance between
them as 68x68 array and use them as feature in the network. 

There exist a confusion between sad/neutral and fear/surprised in this version. With more training data,
it could be possible to distinguish them. 

The network is trained and tested on a mixture of CK+ and JAFFE datasets. The training accuracy is 92.5% where the 
testing accuracy is 84%.

In order to train the network with other dataset, it is required to:
All the images in the dataset are divided into folders structured as follows: (a script for each dataset is
executed to correctly form the directories.)

Under dataset name,
- 0: Contains all images labelled as “angry”.
- 1: Contains all images labelled as “disgust”.
- 2: Contains all images labelled as “fear”.
- 3: Contains all images labelled as “happy”
- 4: Contains all images labelled as “neutral”.
- 5: Contains all images labelled as “sad”.
- 6: Contains all images labelled as “surprised”.

Once we have this structure, “generate_csv.py” is executed with this path to get the csv files with the 
labels and the calculated facial landmark distances.

To download the csv file used in this repository: https://www.dropbox.com/s/bpy7y2erk2bpvgr/JAFFE_CK.csv?dl=0
Place it under "data" folder before running "NN.py".

Used libraries:
- Keras + Tensorflow
- dlib

Datasets:
- CK+
- JAFFE

Training takes a few minutes and it is launched by executing "NN.py".
"demo.py" uses the the pretrained mode by default. To use a custom model, it is required to change 
the variable "usePretrained" to zero and set the path to model and weights. 




