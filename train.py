"""
This file trains the RBM and saves the model to a file
"""
# dataParser is used for reading the MNIST dataset
from dataParser import *
# RBM is used for training a model and saving the model parameters
from rbm import RBM

# Define the RBM structure
rbm_object = RBM(input_size=28*28, hidden_layer_size=100, time_steps=3)

# Read MNIST train image data
data_train = readData('data/train.gz', isTrain=True)
# Read MNIST test image data
data_test = readData('data/test.gz', isTrain=False)

# Train the model
rbm_object.train(data=data_train, learning_rate=5e-3, alpha=0.5, beta=5e-4, epochs=50, batch_size=64)
# Save the model weights
rbm_object.save_weights(filename='model_temp')

# Run sample inference on one of the test images
rbm_object.inference(v=data_test[15, :])
