# tensorflow keras library is used for implementing the softmax classifier
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Activation 

# dataParser is used for reading the MNIST dataset
from dataParser import *

# Define input size, output size, and maximum no. of training iterations
input_size = 784
output_size = 10
epochs = 50

# Read training dataset
[x, y] = readDataset(data_filepath='data/train.gz', label_filepath='data/train-labels.gz', isTrain=True)
# Read test dataset
[test_data, test_label] = readDataset(data_filepath='data/test.gz', label_filepath='data/test-labels.gz', isTrain=False)

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

