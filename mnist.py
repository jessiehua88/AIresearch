import argparse, random, sys
import tensorflow as tf
import numpy as np


# Download dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(FLATS.data_dir, one_hot=TRUE)


def make_ff_layer(size, x):
    x = tf.nn.relu(tf.keras.layers.Dense(size)(x))
    x = tf.nn.relu(tf.keras.layers.Dense(size)(x))
    x = tf.keras.layers.Dense(10)(x)
    return x




