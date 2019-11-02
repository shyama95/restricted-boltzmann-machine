from dataParser import *
from rbm import RBM

import numpy as np

rbm_object = RBM(input_size=28*28, hidden_layer_size=100, time_steps=3)
data_test = readData('data/test.gz', isTrain=False)
rbm_object.load_weights()

rbm_object.inference(v=data_test[15, :])

