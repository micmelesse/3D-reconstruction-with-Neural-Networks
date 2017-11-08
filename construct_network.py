import tensorflow as tf
import numpy as np

# construct encoder


def net_encoder(X):
    ret = [];
    ret.append(X);
    layer_output = tf.layers.conv2d(X, 1, 7)
    ret.append(layer_output)
    for i in range(5):
        conv_output = tf.layers.conv2d(layer_output, 1, 3)
        layer_output = tf.layers.max_pooling2d(conv_output, 1, 3)
        ret.append(layer_output)
    ret.append(tf.contrib.layers.fully_connected(layer_output, 1024));
    return ret;


def net_decoder(X):
    for i in range(5):
        layer_output = tf.layers.conv3d(layer_output, 1, 3)
        layer_output = tf.layers.max_pooling2d(layer_output, 1, 3)
    return tf.nn.softmax(layer)


def net_lstm(X):
    pass


def construct_network(input_tensor):
    pass
