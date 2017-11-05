import tensorflow as tf
import numpy as np

# construct encoder



def encode(X):
    layer_output=tf.layers.conv2d(X,1,7);
    for i in range(5):
        conv_output=tf.layers.conv2d(layer_output,1,3);
        layer_output=tf.layers.max_pooling2d(conv_output,1,3);
    return tf.contrib.layers.fully_connected(layer_output,1024);


def decode(X):
    for i in range(5):
        layer_output=tf.layers.conv3d(layer_output,1,3);
        layer_output=tf.layers.max_pooling2d(layer_output,1,3);
    return tf.nn.softmax(layer)





def lstm:





def construct_network(input_tensor):
