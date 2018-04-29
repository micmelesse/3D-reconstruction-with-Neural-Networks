import tensorflow as tf
import numpy as np
from numpy.random import choice


def fully_connected_sequence(sequence, initializer=None):
    with tf.name_scope("fully_connected_sequence"):
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        weights = tf.Variable(
            init([1024, 1024]), name="weights")
        bias = tf.Variable(init([1024]), name="bias")

        ret = tf.map_fn(lambda a: tf.nn.bias_add(
            tf.matmul(a, weights), bias), sequence, name='fully_connected_map')

        tf.summary.histogram("weights", weights)
        tf.summary.histogram("bias", bias)

    return ret


def max_pool_sequence(sequence, K=[1, 2, 2, 1], S=[1, 2, 2, 1], P="SAME"):
    with tf.name_scope("max_pool_sequence"):
        ret = tf.map_fn(lambda a: tf.nn.max_pool(a, K, S, padding=P),
                        sequence, name="max_pool_map")
    return ret


def relu_sequence(sequence):
    with tf.name_scope("relu_sequence"):
        ret = tf.map_fn(tf.nn.relu, sequence, name="relu_map")
    return ret


def flatten_sequence(sequence):
    with tf.name_scope("flatten_sequence"):
        ret = tf.map_fn(
            tf.contrib.layers.flatten,  sequence, name="flatten_map")
    return ret


def conv_sequence(sequence, fm_count_in, fm_count_out, initializer=None, K=3, S=[1, 1, 1, 1], P="SAME"):
    with tf.name_scope("conv_sequence"):
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        kernel = tf.Variable(
            init([K, K, fm_count_in, fm_count_out]), name="kernel")
        bias = tf.Variable(init([fm_count_out]), name="bias")
        ret = tf.map_fn(lambda a: tf.nn.bias_add(tf.nn.conv2d(
            a, kernel, S, padding=P, name="conv2d"), bias), sequence, name="conv2d_map")

        # feature_map = tf.transpose(tf.expand_dims(
        #     ret[0, 0, :, :, :], -1), [2, 0, 1, 3])
        # tf.summary.image("kernel", kernel)
        # tf.summary.image("feature_map", feature_map)

        tf.summary.histogram("kernel", kernel)
        tf.summary.histogram("bias", bias)
    return ret


def encoder_block(sequence, fm_count_in, fm_count_out,  K=3, initializer=None):
    with tf.name_scope("encoder_block"):
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        conv = conv_sequence(sequence, fm_count_in,
                             fm_count_out, K=K, initializer=init)
        pool = max_pool_sequence(conv)
        relu = relu_sequence(pool)
        return relu


class Simple_Encoder:
    def __init__(self, sequence, feature_map_count=[96, 128, 256, 256, 256, 256], initializer=None):
        with tf.name_scope("Simple_Encoder"):
            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            N = len(feature_map_count)

            # convolution stack
            cur_tensor = encoder_block(
                sequence, 3, feature_map_count[0], K=7, initializer=init)
            for i in range(1, N):
                cur_tensor = encoder_block(
                    cur_tensor, feature_map_count[i-1], feature_map_count[i], initializer=init)

            # final block
            flat = flatten_sequence(cur_tensor)
            fc0 = fully_connected_sequence(flat)
            self.out_tensor = relu_sequence(fc0)


# class Simple_Encoder_old:
#     def __init__(self, prev_layer, feature_map_count=[96, 128, 256, 256, 256, 256]):

#         assert (len(feature_map_count) == 6)
#         self.out_tensor = prev_layer
#         kernel_shape = [3, 3]

#         for i in range(7):
#             if i < 6:
#                 with tf.name_scope("conv_block"):
#                     kernel_shape = [7, 7] if i is 0 else kernel_shape
#                     self.out_tensor = tf.map_fn(lambda a: tf.layers.conv2d(
#                         a, filters=feature_map_count[i], padding='SAME', kernel_size=kernel_shape, activation=None), self.out_tensor, name="conv2_map")
#                     self.out_tensor = tf.map_fn(
#                         lambda a: tf.layers.max_pooling2d(a, 2, 2), self.out_tensor, name="max_pool_map")
#                     self.out_tensor = tf.map_fn(
#                         tf.nn.relu,  self.out_tensor, name="relu_map")
#             elif i == 6:
#                 self.out_tensor = tf.map_fn(
#                     tf.contrib.layers.flatten,  self.out_tensor, name="flatten_map")
#                 self.out_tensor = tf.map_fn(lambda a: tf.contrib.layers.fully_connected(
#                     a, 1024, activation_fn=None), self.out_tensor, name='fully_connected_map')
#                 self.out_tensor = tf.map_fn(
#                     tf.nn.relu,  self.out_tensor, name="relu_map")
