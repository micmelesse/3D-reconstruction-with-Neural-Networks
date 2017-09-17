import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import tensorflow as tf

# setup
sess = tf.Session()

a = tf.constant(1,name='a')
b = tf.constant(2,name='b')
c = a + b
with sess:
    print(c.eval())

writer = tf.summary.FileWriter("./log/")
writer.add_graph(sess.graph);
