
import tensorflow as tf


class Voxel_Softmax:
    def __init__(self, Y, logits):
        with tf.name_scope("loss"):
            # softmax =
            log_softmax = tf.nn.log_softmax(logits)  # avoids log(0)
            label = tf.one_hot(Y, 2)
            cross_entropy = tf.reduce_sum(-tf.multiply(label,
                                                       log_softmax), axis=-1)
            losses = tf.reduce_mean(cross_entropy, axis=[1, 2, 3])
            self.prediction = tf.argmax(tf.nn.softmax(logits), -1)
            self.batch_loss = tf.reduce_mean(losses)
            # tf.summary.scalar("loss", batch_loss)
