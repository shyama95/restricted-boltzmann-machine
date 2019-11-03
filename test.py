"""
This file runs inference on the trained RBM model and visualizes 
the fantasy images
"""
# dataParser is used for reading the MNIST dataset
from dataParser import *
# RBM model is used for running inference on saved model
from rbm import RBM

# Define the RBM structure
rbm_object = RBM(input_size=28*28, hidden_layer_size=100, time_steps=3)
# Read MNIST test image data
data_test = readData('data/test.gz', isTrain=False)
# Load RBM saved model
rbm_object.load_weights(filename='model.npy')

# Run inference for 2 images
rbm_object.inference(v=data_test[100, :])
rbm_object.inference(v=data_test[115, :])

