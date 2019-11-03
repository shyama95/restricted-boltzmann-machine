# tensorflow keras library is used for implementing the softmax classifier
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Activation 
import numpy as np

# RBM model is used for generating the RBM features
from rbm import RBM
# dataParser is used for reading the MNIST dataset
from dataParser import *

# Define input size, output size, and maximum no. of training iterations
input_size = 100
output_size = 10
epochs = 100

# Read training dataset
[x, y] = readDataset(data_filepath='data/train.gz', label_filepath='data/train-labels.gz', isTrain=True)

# Define the RBM structure
rbm_object = RBM(input_size=28*28, hidden_layer_size=100, time_steps=3)
# Load RBM saved model
rbm_object.load_weights(filename='model_temp.npy')

# Extract RBM features from the training data
x = np.dot(x, rbm_object.W) + np.tile(rbm_object.c, (60000, 1))

# Read test dataset
[test_data, test_label] = readDataset(data_filepath='data/test.gz', label_filepath='data/test-labels.gz', isTrain=False)

# Extract RBM features from the test data
test_data = np.dot(test_data, rbm_object.W) + np.tile(rbm_object.c, (10000, 1))

# Split training dataset into training and validation datasets
# 80% data used for training and 20% for validation
train_data = x[0:48000, :]
train_label = y[0:48000, :]

validation_data = x[48000:, :]
validation_label = y[48000:, :]

# Define the softmax regression model
model = Sequential()
model.add(Dense(output_size, activation='softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
history = model.fit(train_data, train_label, nb_epoch=epochs, verbose=1, validation_data=(validation_data, validation_label)) 
# Run inference on test dataset
score = model.evaluate(test_data, test_label, verbose=0)
print("Test accuracy is {}".format(score[1]))

