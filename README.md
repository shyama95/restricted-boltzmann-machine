# Restricted Boltzmann Machine (RBM)
## Abstract
This project implements a Restrictive Boltzmann Machine (RBM) in python using numpy only. It uses Gibbs sampling in Restrictive Boltzmann Machine (RBM) and updates weights using contrastive divergence. The project is tested on MNIST dataset.  
The project also performs a comparison of softmax based digit classfication on MNIST dataset:
* using pixels only as features
* using learned RBM features
## Dependencies
- python v3
- python libraries : gzip, numpy, matplotlib, tensorflow (for softmax classifier)
## Assumptions
* training data is present in data/train.gz
* training labels are present in data/train-labels.gz
* test data is present in data/test.gz
* test labels are present in data/test-labels.gz
## Instructions to run
For training:
~~~~
python3 train.py
~~~~
For visualising fantasy images:
~~~~
python3 test.py
~~~~
For running softmax classifier based on pixels as features:
~~~~
python3 softmax_classifier_pixel_features.py
~~~~
For running softmax classifier based on RBM features:
~~~~
python3 softmax_classifier_rbm_features.py
~~~~
#### dataParser.py is used for parsing the MNIST dataset
## Results
![Sample fantasy image output](https://github.com/shyama95/restricted-boltzmann-machine/blob/master/sample-fantasy-images.png)
