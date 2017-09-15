import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import tensorflow as tf


a = tf.constant(1)
b = tf.constant(2)
c = a + b
with tf.Session():
    print(c.eval())
