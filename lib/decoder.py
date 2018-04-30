import tensorflow as tf


def relu_vox(vox):
    with tf.name_scope("relu_vox"):
        ret = tf.nn.relu(vox, name="relu")
    return ret


def unpool_vox(value):  # from tenorflow github board
    with tf.name_scope('unpool_vox'):
        sh = value.get_shape().as_list()
        dim = len(sh[1: -1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))

        for i in range(dim, 0, -1):
            out = tf.concat([out, tf.zeros_like(out)], i)

        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size)
    return out


def conv_vox(vox, fv_count_in, fv_count_out, K=3, S=[1, 1, 1, 1, 1], initializer=None, P="SAME"):
    with tf.name_scope("conv_vox"):
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        kernel = tf.Variable(
            init([K, K, K, fv_count_in, fv_count_out]), name="kernel")
        bias = tf.Variable(init([fv_count_out]), name="bias")
        ret = tf.nn.bias_add(tf.nn.conv3d(
            vox, kernel, S, padding=P, name="conv3d"), bias)
        tf.summary.histogram("kernel", kernel)
        tf.summary.histogram("bias", bias)

    return ret


def decoder_block(vox, fv_count_in, fv_count_out, K=3, initializer=None, unpool=False):
    with tf.name_scope("decoder_block"):
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        conv = conv_vox(vox, fv_count_in, fv_count_out, initializer=init)
        if unpool:
            return relu_vox(unpool_vox(conv))

    return relu_vox(conv)


class Simple_Decoder:
    def __init__(self, hidden_state, feature_vox_count=[128, 128, 128, 64, 32, 2], initializer=None):
        with tf.name_scope("Simple_Decoder"):
            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            N = len(feature_vox_count)
            cur_tensor = unpool_vox(hidden_state)
            cur_tensor = decoder_block(
                cur_tensor, 128, feature_vox_count[0], K=7, initializer=init)
            for i in range(1, N-1):
                unpool = True if i < 3 else False
                cur_tensor = decoder_block(
                    cur_tensor, feature_vox_count[i-1], feature_vox_count[i], initializer=init, unpool=unpool)

            self.out_tensor = conv_vox(
                cur_tensor, feature_vox_count[-2], feature_vox_count[-1], initializer=init)


# class ConvTranspose_Decoder:
#     pass


# class Simple_Decoder_old:
#     def __init__(self, prev_layer, filter_sizes=[128, 128, 128, 64, 32, 2]):
#         assert (len(filter_sizes) == 6)
#         self.out_tensor = prev_layer
#         kernel_shape = [3, 3, 3]

#         for i in range(6):
#             with tf.name_scope("deconv_block"):
#                 if i == 0:
#                     self.out_tensor = unpooling3d(self.out_tensor)
#                 elif i in range(1, 3):  # scale up hidden state to 32*32*32
#                     self.out_tensor = tf.layers.conv3d(
#                         self.out_tensor, padding='SAME', filters=filter_sizes[i], kernel_size=kernel_shape, activation=None)
#                     self.out_tensor = unpooling3d(self.out_tensor)
#                     self.out_tensor = tf.nn.relu(self.out_tensor)
#                 elif i in range(3, 5):  # reduce number of channels to 2
#                     self.out_tensor = tf.layers.conv3d(
#                         self.out_tensor, padding='SAME', filters=filter_sizes[i], kernel_size=kernel_shape, activation=None)
#                     self.out_tensor = tf.nn.relu(self.out_tensor)
#                 elif i == 5:  # final conv before softmax
#                     self.out_tensor = tf.layers.conv3d(
#                         self.out_tensor, padding='SAME', filters=filter_sizes[i], kernel_size=kernel_shape, activation=None)
