# -*- coding: utf-8 -*-

import  tensorflow as tf
from    tensorflow.keras import layers, optimizers, datasets, Sequential
import  os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

x = tf.random.normal([4, 80, 10])

xt0 = x[:, 0, :]

print(xt0.shape)

cell = tf.keras.layers.SimpleRNNCell(64)
cell1 = tf.keras.layers.SimpleRNNCell(64)
state0 = [tf.zeros([4, 64])]
state1 = [tf.zeros([4, 64])]

out0, state0 = cell(xt0, state0)
out1, state1 = cell1(out0, state1)

print(out1.shape, state1[0].shape)