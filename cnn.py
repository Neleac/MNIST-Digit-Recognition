'''
Author: Caelen Wang

Overview:
Convolutional Neural Network for handwritten digit classification. Desgined for the MNIST dataset. 

Requirements:
	Python 3.6
	Pandas library
	Keras library
	TensorFlow backend

Usage Details:
Change the variables to specify the training dataset and the testing dataset.
Note that the training set MUST have the labels in the first column.
Upon running the script, the model will train based on given training data.

Output:
	cnn_arch.json (stores the network architecture)
	cnn_weights.h5 (stores the weights associated with the neurons)
	predictions.csv (stores predictions made on given testing data)
'''


#import required libraries
import pandas as pd
import numpy as np

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D


#constants
ROW_DIM = 28
COL_DIM = 28
CHANNELS = 1
OUTPUT_DIM = 10


#variables (CHANGE ME)
training_data = 'train.csv'
testing_data = 'test.csv'


#reshapes dataset to required dimensions
def reshape(data):
    return data.reshape(data.shape[0], ROW_DIM, COL_DIM, CHANNELS)


#load training data into pandas dataframe
df_train = pd.read_csv(training_data)

#split into features and target, convert to numpy arrays
target = df_train.iloc[:, 0].values
features = df_train.iloc[:, 1:].values

#one-hot encode target , reshape features
target = to_categorical(target)
features = reshape(features)


'''
build neural network
	Sequential architecture
	2 convolution layers, 2 pooling layers
	dropout layer to avoid overfitting 
	flatten layer converts to vector
	3 fully connected layers
'''
model = Sequential()
model.add(Conv2D(30, (5, 5), input_shape = (ROW_DIM, COL_DIM, CHANNELS), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(15, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(OUTPUT_DIM, activation = 'softmax'))

#compile neural network
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#train neural network
model.fit(features, target, epochs = 10, batch_size = 200, validation_split = 0.1, verbose = True)


#store model architecture and weights
with open('cnn_arch.json', 'w') as fout:
    fout.write(model.to_json())
model.save_weights('cnn_weights.h5', overwrite=True)

#make predictions on testing data
df_test = pd.read_csv(testing_data).values
df_test = reshape(df_test)
predictions = model.predict_classes(df_test)
pd.DataFrame({"ImageId": list(range(1, len(predictions) + 1)), "Label": predictions}).to_csv('predictions.csv', index = False, header = True)
