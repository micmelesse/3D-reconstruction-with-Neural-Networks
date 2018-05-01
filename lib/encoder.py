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


def conv_sequence(sequence, fm_count_in, fm_count_out, initializer=None, K=3, S=[1, 1, 1, 1], D=[1, 1, 1, 1], P="SAME"):
    with tf.name_scope("conv_sequence"):
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        kernel = tf.Variable(
            init([K, K, fm_count_in, fm_count_out]), name="kernel")
        bias = tf.Variable(init([fm_count_out]), name="bias")
        ret = tf.map_fn(lambda a: tf.nn.bias_add(tf.nn.conv2d(
            a, kernel, S, padding=P, dilations=D, name="conv2d"), bias), sequence, name="conv2d_map")

        # feature_map = tf.transpose(tf.expand_dims(
        #     ret[0, 0, :, :, :], -1), [2, 0, 1, 3])
        # tf.summary.image("kernel", kernel)
        # tf.summary.image("feature_map", feature_map)

        tf.summary.histogram("kernel", kernel)
        tf.summary.histogram("bias", bias)
    return ret


def simple_encoder_block(sequence, fm_count_in, fm_count_out,  K=3, D=[1, 1, 1, 1], initializer=None):
    with tf.name_scope("simple_encoder_block"):
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        conv = conv_sequence(sequence, fm_count_in,
                             fm_count_out, K=K, D=D, initializer=init)
        pool = max_pool_sequence(conv)
        relu = relu_sequence(pool)
        return relu


def residual_encoder_block(sequence, fm_count_in, fm_count_out,  K_1=3, K_2=3, D=[1, 1, 1, 1], initializer=None):
    with tf.name_scope("residual_encoder_block"):
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        conv1 = conv_sequence(sequence, fm_count_in,
                              fm_count_out, K=K_1, D=D, initializer=init)
        relu1 = relu_sequence(conv1)
        conv2 = conv_sequence(relu1, fm_count_out,
                              fm_count_out, K=K_2, D=D, initializer=init)
        relu2 = relu_sequence(conv2)
        out = sequence + relu2

        pool = max_pool_sequence(out)
        return pool


class Residual_Encoder:
    def __init__(self, sequence, feature_map_count=[96, 128, 256, 256, 256, 256], initializer=None):
        with tf.name_scope("Simple_Encoder"):
            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            N = len(feature_map_count)

            # convolution stack
            cur_tensor = residual_block(
                sequence, 3, feature_map_count[0], K_1=7, K_2=3, initializer=init)
            for i in range(1, N):
                cur_tensor = simple_encoder_block(
                    cur_tensor, feature_map_count[i-1], feature_map_count[i], initializer=init)

            # final block
            flat = flatten_sequence(cur_tensor)
            fc0 = fully_connected_sequence(flat)
            self.out_tensor = relu_sequence(fc0)


class Dilated_Encoder:
    def __init__(self, sequence, feature_map_count=[96, 128, 256, 256, 256, 256], initializer=None):
        with tf.name_scope("Simple_Encoder"):
            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            N = len(feature_map_count)

            # convolution stack
            cur_tensor = simple_encoder_block(
                sequence, 3, feature_map_count[0], K=7, D=[1, 2, 2, 1], initializer=init)
            for i in range(1, N):
                cur_tensor = simple_encoder_block(
                    cur_tensor, feature_map_count[i-1], feature_map_count[i], D=[1, 2, 2, 1], initializer=init)

            # final block
            flat = flatten_sequence(cur_tensor)
            fc0 = fully_connected_sequence(flat)
            self.out_tensor = relu_sequence(fc0)


class Simple_Encoder:
    def __init__(self, sequence, feature_map_count=[96, 128, 256, 256, 256, 256], initializer=None):
        with tf.name_scope("Simple_Encoder"):
            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            N = len(feature_map_count)

            # convolution stack
            cur_tensor = simple_encoder_block(
                sequence, 3, feature_map_count[0], K=7, initializer=init)
            for i in range(1, N):
                cur_tensor = simple_encoder_block(
                    cur_tensor, feature_map_count[i-1], feature_map_count[i], initializer=init)

            # final block
            flat = flatten_sequence(cur_tensor)
            fc0 = fully_connected_sequence(flat)
            self.out_tensor = relu_sequence(fc0)
