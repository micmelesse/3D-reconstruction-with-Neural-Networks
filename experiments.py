# %%
# core imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import tensorflow as tf
# %%
# import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# %%
# setup graph
sess = tf.Session()
a = tf.constant(5, name='alpha')
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name='weights')
b = tf.Variable(tf.zeros([1]), name='biases')
# %%
# run graph
with sess:
    print(a.eval())
# %%
# serialize graph for tensorboard
writer = tf.summary.FileWriter("./log/")
writer.add_graph(sess.graph)
