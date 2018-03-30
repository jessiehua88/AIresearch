import argparse, random, sys
#import matplotlib.pyplot as plt

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


def get_tf_session(gpumem):
    """ Returning a session. Set options here if desired. """
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1,
                               intra_op_parallelism_threads=1)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpumem)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    def get_available_gpus():
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session

session = get_tf_session(0.5)
# Define computational graph
x_tf = tf.placeholder(tf.float32, [None, 784])
y_tf = tf.placeholder(tf.float32, [None, 10])


x_1 = tf.nn.relu(tf.keras.layers.Dense(200)(x_tf))
x_2 = tf.nn.relu(tf.keras.layers.Dense(200)(x_1))
y_logits = tf.keras.layers.Dense(10)(x_2)

#y_logits = make_ff_layer(x_tf)
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_tf, logits=y_logits)
)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

correct_prediction = tf.equal(tf.argmax(y_logits,1), tf.argmax(y_tf,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize session
session.run(tf.global_variables_initializer())

## Debugging
#variables = tf.trainable_variables()
#print("\nHere are the variables in our network:")
#params = 0
#for item in variables:
#    print(item)
#    params += np.prod(item.get_shape().as_list())
#    print("Total net parameters: {}\n".format(params))
#sys.exit()


# Feed in data to train network.
num_minibatches = int(mnist.train.images.shape[0] / args.batch)

for epoch in range(100):
    for minibatch in range(num_minibatches):
        x, y = mnist.train.next_batch(args.batch)
        feed = {x_tf: x, y_tf: y}
        _,ce_cost = session.run([train_step, cost], feed_dict=feed)

    feed_t = {
        x_tf: mnist.test.images,
        y_tf: mnist.test.labels,
    }
    acc = session.run(accuracy, feed_dict=feed_t)
    print("epoch {}, accuracy {}".format(epoch, acc))
