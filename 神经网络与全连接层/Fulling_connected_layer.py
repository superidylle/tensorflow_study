# -*- coding: utf-8 -*-

import tensorflow as tf
from   tensorflow import keras
import  os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.random.normal([4, 784])

net = tf.keras.layers.Dense(512)
out = net(x)

print(out.shape)

print(net.kernel.shape)
print(net.bias.shape)

x = tf.random.normal([2, 3])

model = keras.Sequential([
        keras.layers.Dense(2, activation='relu'),
        keras.layers.Dense(2, activation='relu'),
        keras.layers.Dense(2)
    ])

model.build(input_shape=[None, 3])
model.summary()

for p in model.trainable_variables:

    print(p.name, p.shape)