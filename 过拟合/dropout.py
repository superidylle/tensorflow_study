# -*- coding: utf-8 -*-


import tensorflow as tf
from   tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from   tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


"""

model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),  # [b, 784] => [b, 256]
    layers.Dropout(0.5), # 0.5 rate to drop
    layers.Dense(128, activation=tf.nn.relu),  # [b, 256] => [b, 128]
    layers.Dropout(0.5),
    layers.Dense(64, activation=tf.nn.relu),  # [b, 128] => [b, 64]
    layers.Dense(32, activation=tf.nn.relu),  # [b, 64] => [b, 32]
    layers.Dense(10)  # [b, 32] => [b, 10]

"""


