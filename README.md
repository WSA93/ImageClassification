# ImageClassification
This program is an image classification script which uses convolutional neural network present in Keras library. This script retrained an already trained model known as vgg16.

The model is trained on three categories of images purse, watches and shirts by removing the last layer of vgg16 model and adding these three layers.

Flask server is used to host the trained model and prediction is done by uploading the image on a webpage. The image is sent to the flask server and prediction is returned to the webpage as the name of the category.  
