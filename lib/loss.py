
import tensorflow as tf


class Voxel_Softmax:
    def __init__(self, Y, logits):
        with tf.name_scope("Loss_Voxel_Softmax"):
            label = Y
            epsilon = 1e-10
            self.softmax = tf.clip_by_value(
                tf.nn.softmax(logits), epsilon, 1-epsilon)
            log_softmax = tf.log(self.softmax)
            # log_softmax = tf.nn.log_softmax(self.logits)  # avoids log(0)
            # label = tf.one_hot(Y, 2) # one hot encoding is done in preprocessing
            cross_entropy = tf.reduce_sum(-tf.multiply(label,
                                                       log_softmax), axis=-1)
            losses = tf.reduce_mean(cross_entropy, axis=[1, 2, 3])
            self.loss = tf.reduce_mean(losses)
