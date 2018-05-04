import tensorflow as tf
import numpy as np
from lib import utils
from numpy.random import choice


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

        # visualization code
        params = utils.read_params()
        image_count = params["VIS"]["IMAGE_COUNT"]
        if params["VIS"]["KERNELS"]:
            kern_1 = tf.concat(tf.unstack(kernel, axis=-1), axis=-1)
            kern_2 = tf.transpose(kern_1, [2, 0, 1])
            kern_3 = tf.expand_dims(kern_2, -1)
            tf.summary.image("2d kernel", kern_3, max_outputs=image_count)

        if params["VIS"]["FEATURE_MAPS"]:
            feature_map_1 = tf.concat(tf.unstack(ret, axis=4), axis=2)
            feature_map_2 = tf.concat(
                tf.unstack(feature_map_1, axis=1), axis=2)
            feature_map_3 = tf.expand_dims(feature_map_2, -1)
            tf.summary.image("feature_map", feature_map_3,
                             max_outputs=image_count)

        if params["VIS"]["HISTOGRAMS"]:
            tf.summary.histogram("kernel", kernel)
            tf.summary.histogram("bias", bias)

        if params["VIS"]["SHAPES"]:
            print(ret.shape)
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

        params = utils.read_params()
        if params["VIS"]["HISTOGRAMS"]:
            tf.summary.histogram("weights", weights)
            tf.summary.histogram("bias", bias)

    return ret


def flatten_sequence(sequence):
    with tf.name_scope("flatten_sequence"):
        ret = tf.map_fn(
            tf.contrib.layers.flatten,  sequence, name="flatten_map")
    return ret


def block_simple_encoder(sequence, fm_count_in, fm_count_out,  K=3, D=[1, 1, 1, 1], initializer=None):
    with tf.name_scope("block_simple_encoder"):
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        conv = conv_sequence(sequence, fm_count_in,
                             fm_count_out, K=K, D=D, initializer=init)
        pool = max_pool_sequence(conv)
        relu = relu_sequence(pool)
    return relu


def block_residual_encoder(sequence, fm_count_in, fm_count_out,  K_1=3, K_2=3, K_3=1, D=[1, 1, 1, 1], initializer=None, pool=True):
    with tf.name_scope("block_residual_encoder"):
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        out = sequence
        if K_1 != 0:
            conv1 = conv_sequence(out, fm_count_in,
                                  fm_count_out, K=K_1, D=D, initializer=init)
            relu1 = relu_sequence(conv1)
            out = relu1

        if K_2 != 0:
            conv2 = conv_sequence(out, fm_count_out,
                                  fm_count_out, K=K_2, D=D, initializer=init)
            relu2 = relu_sequence(conv2)
            out = relu2

        if K_3 != 0:
            conv3 = conv_sequence(out, fm_count_out,
                                  fm_count_out, K=K_3, D=D, initializer=init)
            out = conv3 + relu2

        if pool:
            pool = max_pool_sequence(out)
            out = pool

        return out


class Residual_Encoder:
    def __init__(self, sequence, feature_map_count=[96, 128, 256, 256, 256, 256], initializer=None):
        with tf.name_scope("Residual_Encoder"):
            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            cur_tensor = block_residual_encoder(
                sequence, 3, feature_map_count[0], K_1=7, K_2=3, K_3=0, initializer=init)
            # convolution stack
            N = len(feature_map_count)
            for i in range(1, N):
                cur_tensor = block_residual_encoder(
                    cur_tensor, feature_map_count[i-1], feature_map_count[i], initializer=init)

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
            cur_tensor = block_simple_encoder(
                sequence, 3, feature_map_count[0], K=7, initializer=init)
            for i in range(1, N):
                cur_tensor = block_simple_encoder(
                    cur_tensor, feature_map_count[i-1], feature_map_count[i], initializer=init)

            # final block
            flat = flatten_sequence(cur_tensor)
            fc0 = fully_connected_sequence(flat)
            self.out_tensor = relu_sequence(fc0)


class Dilated_Encoder:
    def __init__(self, sequence, feature_map_count=[96, 128, 256, 256, 256, 256], initializer=None):
        with tf.name_scope("Dilated_Encoder"):
            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            N = len(feature_map_count)

            # convolution stack
            cur_tensor = block_simple_encoder(
                sequence, 3, feature_map_count[0], K=7, D=[1, 2, 2, 1], initializer=init)
            for i in range(1, N):
                cur_tensor = block_simple_encoder(
                    cur_tensor, feature_map_count[i-1], feature_map_count[i], D=[1, 2, 2, 1], initializer=init)

            # final block
            flat = flatten_sequence(cur_tensor)
            fc0 = fully_connected_sequence(flat)
            self.out_tensor = relu_sequence(fc0)
