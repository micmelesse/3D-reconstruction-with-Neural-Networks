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





def conv_voxel(sequence, n_in_filter, n_out_filter, initializer=None, K=3, S=[1, 1, 1, 1], P="SAME"):
    with tf.name_scope("conv_voxel"):
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        kernel = tf.Variable(
            init([K, K, n_in_filter, n_out_filter]), name="kernel")
        bias = tf.Variable(init([n_out_filter]), name="bias")
        ret = tf.map_fn(lambda a: tf.nn.bias_add(tf.nn.conv2d(
            a, kernel, S, padding=P), bias), sequence, name="conv_voxel")

        feature_map = tf.transpose(tf.expand_dims(
            ret[0, 0, :, :, :], -1), [2, 0, 1, 3])

        # tf.summary.image("kernel", kernel)
        tf.summary.image("feature_map", feature_map)
        tf.summary.histogram("kernel", kernel)
        tf.summary.histogram("bias", bias)

    return ret


def fully_connected_voxel(sequence, initializer=None):
    with tf.name_scope("fully_connected_voxel"):
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        weights = tf.Variable(
            init([1024, 1024]), name="weights")
        bias = tf.Variable(init([1024]), name="bias")

        ret = tf.map_fn(lambda a: tf.nn.bias_add(
            tf.matmul(a, weights), bias), sequence, name='fully_connected_voxel')

        tf.summary.histogram("weights", weights)
        tf.summary.histogram("bias", bias)

    return ret


def max_pool_voxel(sequence, K=[1, 2, 2, 1], S=[1, 2, 2, 1], P="SAME"):
    with tf.name_scope("max_pool_voxel"):
        ret = tf.map_fn(lambda a: tf.nn.max_pool(a, K, S, padding=P),
                        sequence, name="max_pool_voxel")
    return ret


def relu_voxel(sequence):
    with tf.name_scope("relu_voxel"):
        ret = tf.map_fn(tf.nn.relu, sequence, name="relu_voxel")
    return ret


def flatten_voxel(sequence):
    with tf.name_scope("flatten_voxel"):
        ret = tf.map_fn(
            tf.contrib.layers.flatten,  sequence, name="flatten_voxel")
    return ret


class Simple_Encoder:
    def __init__(self, sequence, feature_map_count=[3, 96, 128, 256, 256, 256, 256], initializer=None):
        assert (len(feature_map_count) == 7)
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        # sequence = tf.transpose(sequence, [1, 0, 2, 3, 4])
        # block 0
        conv0 = conv_voxel(
            sequence, feature_map_count[0], feature_map_count[1], K=7, initializer=init)
        pool0 = max_pool_voxel(conv0)
        relu0 = relu_voxel(pool0)

        # block 1
        conv1 = conv_voxel(
            relu0, feature_map_count[1], feature_map_count[2], initializer=init)
        pool1 = max_pool_voxel(conv1)
        relu1 = relu_voxel(pool1)

        # block 2
        conv2 = conv_voxel(
            relu1, feature_map_count[2], feature_map_count[3], initializer=init)
        pool2 = max_pool_voxel(conv2)
        relu2 = relu_voxel(pool2)

        # block 3
        conv3 = conv_voxel(
            relu2, feature_map_count[3], feature_map_count[4], initializer=init)
        pool3 = max_pool_voxel(conv3)
        relu3 = relu_voxel(pool3)

        # block 4
        conv4 = conv_voxel(
            relu3, feature_map_count[4], feature_map_count[5], initializer=init)
        pool4 = max_pool_voxel(conv4)
        relu4 = relu_voxel(pool4)

        # block 4
        conv5 = conv_voxel(
            relu4, feature_map_count[5], feature_map_count[6], initializer=init)
        pool5 = max_pool_voxel(conv5)
        relu6 = relu_voxel(pool5)

        # final block
        flat = flatten_voxel(relu6)
        fc0 = fully_connected_voxel(flat)
        relu7 = relu_voxel(fc0)
        # self.out_tensor = tf.transpose(relu7, [1, 0, 2])
        self.out_tensor = relu7


class ConvTranspose_Decoder:
    pass


class Simple_Decoder_old:
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
