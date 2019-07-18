# -*- coding: utf-8 -*-

import tensorflow as tf
from   tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from   tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def preprocess(x, y):
    # [0-255] => [-1, 1]
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

batchsz = 128

(x, y), (x_test, y_test) = datasets.cifar10.load_data()

x_train, x_val = tf.split(x, num_or_size_splits=[50000, 10000])
y_train, y_val = tf.split(y, num_or_size_splits=[50000, 10000])

db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.map(preprocess).shuffle(50000).batch(batchsz)

db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
db_val = db_val.map(preprocess).shuffle(10000).batch(batchsz)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(batchsz)

"""
network = MyNetwork()
network.compile(optimizer=optimizers.Adam(lr=1e-3),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

network.fit(db_train, epochs=15, validation_data=db_val, validation_freq=1)

network.evaluate(db_test)

"""

# method 1: random selective the training set and validation set
for epoch in range(500):

    idx = tf.range(60000)
    idx = tf.random.shuffle(idx)
    x_train, y_train = tf.gather(x, idx[:50000]), tf.gather(y, idx[:50000])
    x_val, y_val = tf.gather(x, idx[-10000:]), tf.gather(y, idx[-10000:])

    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    db_train = db_train.map(preprocess).shuffle(50000).batch(batchsz)

    db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    db_val = db_val.map(preprocess).shuffle(10000).batch(batchsz)

# method 2:

"""

network.fit(db_train_val, epochs=15, validation_split=0.1, validation_freq=1)

"""



