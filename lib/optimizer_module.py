import tensorflow as tf


class SGD_optimizer:
    def __init__(self, loss, learn_rate):
        with tf.name_scope("update"):
            self.step_count = tf.Variable(
                0, trainable=False, name="step_count")

            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=learn_rate)

            grads_and_vars = optimizer.compute_gradients(loss)

            self.apply_grad = optimizer.apply_gradients(
                grads_and_vars, global_step=self.step_count)
