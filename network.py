import os
import sys
import math
import utils
import random
import layers
import dataset
import binvox_rw
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
# tf.contrib.eager.enable_eager_execution()


def setup_net():
    # place holders
    x = tf.placeholder(tf.float32, [None, 24, 137, 137, 3])
    y = tf.placeholder(tf.float32, [None, 32, 32, 32])

    # encoder network
    cur_tensor = x
    encoder_outputs = [cur_tensor]
    print(cur_tensor.shape)
    k_s = [3, 3]
    conv_filter_count = [96, 128, 256, 256, 256, 256]
    for i in range(6):
        ks = [7, 7]if i is 0 else k_s
        with tf.name_scope("encoding_block"):
            cur_tensor = tf.map_fn(lambda a: tf.layers.conv2d(
                a, filters=conv_filter_count[i], padding='SAME', kernel_size=k_s, activation=None), cur_tensor)
            cur_tensor = tf.map_fn(
                lambda a: tf.layers.max_pooling2d(a, 2, 2), cur_tensor)
            cur_tensor = tf.map_fn(tf.nn.relu, cur_tensor)
            print(cur_tensor.shape)
            encoder_outputs.append(cur_tensor)

    # flatten tensors
    cur_tensor = tf.map_fn(tf.contrib.layers.flatten, cur_tensor)
    cur_tensor = tf.map_fn(lambda a: tf.contrib.layers.fully_connected(
        a, 1024, activation_fn=None), cur_tensor)
    encoder_outputs.append(cur_tensor)
    print(cur_tensor.shape)

    # recurrent module
    recurrent_module = layers.GRU_R2N2()

    # prepare input
    cur_tensor = encoder_outputs[-1]
    stacked_input = cur_tensor
    for i in range(3):
        stacked_input = tf.stack([stacked_input] * 4, axis=0)
    print(stacked_input.shape)

    # initial hidden state
    hidden_state = tf.zeros_like(stacked_input[:, :, :, :, 0, 0:256])

    # feed batches of seqeuences
    for t in range(24):
        input_frames = stacked_input[:, :, :, :, t, :]
        hidden_state = recurrent_module.call(input_frames, hidden_state)
    print(hidden_state.shape)

    # decoding network
    cur_tensor = tf.transpose(hidden_state, [3, 0, 1, 2, 4])
    print(cur_tensor.shape)

    decoder_outputs = [cur_tensor]
    cur_tensor = layers.unpool3D(cur_tensor)
    print(cur_tensor.shape)
    decoder_outputs.append(cur_tensor)

    k_s = [3, 3, 3]
    deconv_filter_count = [128, 128, 128, 64, 32, 2]
    for i in range(2, 4):
        with tf.name_scope("decoding_block"):
            cur_tensor = tf.layers.conv3d(
                cur_tensor, padding='SAME', filters=deconv_filter_count[i], kernel_size=k_s, activation=None)
            cur_tensor = layers.unpool3D(cur_tensor)
            cur_tensor = tf.nn.relu(cur_tensor)
            print(cur_tensor.shape)
            decoder_outputs.append(cur_tensor)

    for i in range(4, 6):
        with tf.name_scope("decoding_block_without_unpooling"):
            cur_tensor = tf.layers.conv3d(
                cur_tensor, padding='SAME', filters=deconv_filter_count[i], kernel_size=k_s, activation=None)
            cur_tensor = tf.nn.relu(cur_tensor)
            print(cur_tensor.shape)
            decoder_outputs.append(cur_tensor)

    # 3d voxel-wise softmax
    y_hat = tf.nn.softmax(decoder_outputs[-1])
    p = y_hat[:, :, :, :, 0]
    q = y_hat[:, :, :, :, 1]
    cross_entropies = tf.reduce_sum(-tf.multiply(tf.log(p), y) -
                                    tf.multiply(tf.log(q), 1 - y), [1, 2, 3])
    loss = tf.reduce_mean(cross_entropies)
    optimizer = tf.train.GradientDescentOptimizer(1).minimize(loss)
