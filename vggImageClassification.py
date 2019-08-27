
import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation 
from keras.layers.core import Dense , Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from keras.models import load_model

train_path = 'E:/ML/CNN/dataset/train'
valid_path = 'E:/ML/CNN/dataset/valid'

 
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224),classes=['purse','shirt','watch'], batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224),classes=['purse','shirt','watch'], batch_size=4)




#vgg built-------------------------------------------

vgg16_model = keras.applications.vgg16.VGG16()

vgg16_model.summary()

vgg16_model.layers.pop()

vgg16_model.summary()

type(vgg16_model)

vggModel = Sequential()

for layer in vgg16_model.layers:
    vggModel.add(layer)

vggModel.summary()

for layer in vggModel.layers:
    layer.trainable = False
    
vggModel.add(Dense(3,activation='softmax'))

vggModel.summary()


#vgg train-------------------------------------

vggModel.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

vggModel.fit_generator(train_batches, steps_per_epoch=44, validation_data=valid_batches, validation_steps=44, epochs=5, verbose=2)



#vgg save-------------------------------------


vggModel.save("E:/ML/CNN/vgg.h5") 

