# -*- coding: utf-8 -*-


import tensorflow as tf
from   tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from   tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




"""
Method 1:

optimizer = SGD(learning_rate = 0.2)

for epoch in range(100):

    # get loss
    
    # chagne learning rate
    
    optimizer.learning_rate = 0.2 * (100 - epoch) / 100
    
    # update weights


"""