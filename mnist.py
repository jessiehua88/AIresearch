import argparse, random, sys
import tensorflow as tf
import numpy as np

# Download dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(FLATS.data_dir, one_hot=TRUE)

# Hyperparmeters
parser = argparse.ArgumentParser()
parser.add_argument('--batch', type = int. default = 100)
parser.add_argument('--layers', type = int. default = 3)
args = parser.parse_args()
print("Our arguments:\n{}".format(args))

# Constructing layer
def make_ff_layer(size, x):
    for i in range(1, args.layers):
        x = tf.nn.relu(tf.keras.layers.Dense(size)(x))
    x = tf.keras.layers.Dense(10)(x)
    return x







