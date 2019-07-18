# -*- coding: utf-8 -*-

import tensorflow as tf
from   tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from   tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




"""
Method 1:

L2_model = keras.models.Sequential([
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

"""

"""

Method 2:

for step, (x, y) in enumerate(db):

    with tf.GradientTape() as tape:

        loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, out, from_logits=True))

        loss_regularization = []
        for p in network.trainable_variables:
            loss_regularization.append(tf.nn.l2_loss(p))

        loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))

        loss = loss + 0.001 * loss_regularization

    grads = tape.gradient(loss, network.trainable_variables)
    optimizers.apply_gradients(zip(grads, network.trainable_variables))

"""


