import argparse, random, sys
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np

# Download dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Hyperparmeters
parser = argparse.ArgumentParser()
parser.add_argument('--batch', type = int, default=100)
parser.add_argument('--layers', type = int, default=3)
parser.add_argument('--size', type = int, default=784)
args = parser.parse_args()
print("Our arguments:\n{}".format(args))

# Constructing layer
def make_ff_layer(size, x):
    for i in range(1, args.layers):
        x = tf.nn.relu(tf.keras.layers.Dense(size)(x))
    x = tf.keras.layers.Dense(10)(x)
    return x

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
x, y = mnist.train.next_batch(args.batch)
y_logits = make_ff_layer(args.size, x)
y_softmax = tf.nn.softmax(y_logits)

plt.plot(y, y_softmax)




