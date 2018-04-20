import tensorflow as tf


def unpooling3d(value):
    with tf.name_scope('unpooling3d'):
        sh = value.get_shape().as_list()
        dim = len(sh[1: -1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))

        for i in range(dim, 0, -1):
            out = tf.concat([out, tf.zeros_like(out)], i)

        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size)
    return out


class Conv_Decoder:
    def __init__(self, prev_layer, filter_sizes=[128, 128, 128, 64, 32, 2]):
        assert (len(filter_sizes) == 6)
        self.out_tensor = prev_layer
        kernel_shape = [3, 3, 3]

        for i in range(6):
            with tf.name_scope("deconv_block"):
                if i == 0:
                    self.out_tensor = unpooling3d(self.out_tensor)
                elif i in range(1, 3):  # scale up hidden state to 32*32*32
                    self.out_tensor = tf.layers.conv3d(
                        self.out_tensor, padding='SAME', filters=filter_sizes[i], kernel_size=kernel_shape, activation=None)
                    self.out_tensor = tf.nn.relu(self.out_tensor)
                    self.out_tensor = unpooling3d(self.out_tensor)
                elif i in range(3, 5):  # reduce number of channels to 2
                    self.out_tensor = tf.layers.conv3d(
                        self.out_tensor, padding='SAME', filters=filter_sizes[i], kernel_size=kernel_shape, activation=None)
                    self.out_tensor = tf.nn.relu(self.out_tensor)
                elif i == 5:  # final conv before softmax
                    self.out_tensor = tf.layers.conv3d(
                        self.out_tensor, padding='SAME', filters=filter_sizes[i], kernel_size=kernel_shape, activation=None)
