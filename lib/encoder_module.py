import tensorflow as tf


class Conv_Encoder:
    def __init__(self, prev_layer, filter_sizes=[96, 128, 256, 256, 256, 256]):

        assert (len(filter_sizes) == 6)
        self.out_tensor = prev_layer
        kernel_shape = [3, 3]

        for i in range(7):
            if i < 6:
                with tf.name_scope("conv_block"):
                    kernel_shape = [7, 7] if i is 0 else kernel_shape
                    self.out_tensor = tf.map_fn(lambda a: tf.layers.conv2d(
                        a, filters=filter_sizes[i], padding='SAME', kernel_size=kernel_shape, activation=None),   self.out_tensor)
                    self.out_tensor = tf.map_fn(
                        lambda a: tf.layers.max_pooling2d(a, 2, 2),  self.out_tensor)
                    self.out_tensor = tf.map_fn(
                        tf.nn.relu,  self.out_tensor)
            elif i == 6:
                self.out_tensor = tf.map_fn(
                    tf.contrib.layers.flatten,  self.out_tensor)
                self.out_tensor = tf.map_fn(lambda a: tf.contrib.layers.fully_connected(
                    a, 1024, activation_fn=None), self.out_tensor)
                self.out_tensor = tf.map_fn(
                    tf.nn.relu,  self.out_tensor)

        with tf.name_scope("verify"):
            self.out_tensor = tf.verify_tensor_all_finite(
                self.out_tensor, "encoder output has Nans or Infs")
