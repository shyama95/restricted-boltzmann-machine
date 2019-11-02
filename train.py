from dataParser import *
from rbm import RBM

import numpy as np

rbm_object = RBM(input_size=28*28, hidden_layer_size=100, time_steps=3)

data_train = readData('data/train.gz', isTrain=True)

rbm_object.train(data=data_train[0:60000, :], learning_rate=0.003, epochs=50)
rbm_object.save_weights()

rbm_object.inference(v=data_test[15, :])
