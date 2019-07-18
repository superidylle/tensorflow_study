# -*- coding: utf-8 -*-

import  tensorflow as tf
from    tensorflow.keras import layers, optimizers, datasets, Sequential
import  os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

x = tf.random.normal([4, 80, 10])

xt0 = x[:, 0, :]

print(xt0.shape)

cell = tf.keras.layers.SimpleRNNCell(64)

# cell.build(input_shape=(None, 4))
# print(cell.trainable_variables)

"""
[      <tf.Variable 'kernel:0'           shape=(4, 64) dtype=float32, >, 
       <tf.Variable 'recurrent_kernel:0' shape=(64, 64) dtype=float32>, 
       <tf.Variable 'bias:0'             shape=(64,) dtype=float32>
]


"""

# xt @ wxh + ht @ whh
# [batch, feature len] @ [feature len, hidden len] +
# [bathc, hidden len] @ [hidden len, hidden len]


# out, hi1 = call(x, h0)
# x: [b, seq len, word vec]
# h0 / h1: [b, h dim]
# out: [b, h dim]

out, ht1= cell(xt0, [tf.zeros([4, 64])])

# out: [4, 64]
#
# print(out.shape, ht1[0].shape)
print(out.shape, ht1[0].shape)
print(id(out), id(ht1[0]))
print(cell.trainable_variables)


