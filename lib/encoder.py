import tensorflow as tf


def conv_sequence(sequence, n_in_filter, n_out_filter, initializer=None, K=3, S=[1, 1, 1, 1], P="SAME"):
    with tf.name_scope("conv_sequence"):
        if initializer is None:
            initializer = tf.random_normal_initializer()

        kernel = tf.Variable(
            initializer([K, K, n_in_filter, n_out_filter]), name="kernel")
        ret = tf.map_fn(lambda a: tf.nn.conv2d(
            a, kernel, S, padding=P), sequence, name="conv_sequence")

    return ret


def max_pool_sequence(sequence, K=[1, 2, 2, 1], S=[1, 1, 1, 1], P="SAME"):
    with tf.name_scope("max_pool_sequence"):
        ret = tf.map_fn(lambda a: tf.nn.max_pool(a, K, S, padding=P),
                        sequence, name="max_pool_sequence")
    return ret


def relu_sequence(sequence):
    with tf.name_scope("relu_sequence"):
        ret = tf.map_fn(tf.nn.relu, sequence, name="relu_sequence")
    return ret


def flatten_sequence(sequence):
    with tf.name_scope("flatten_sequence"):
        flat = tf.map_fn(
            tf.contrib.layers.flatten,  sequence, name="flatten_sequence")

        ret = tf.map_fn(lambda a: tf.contrib.layers.fully_connected(
            a, 1024, activation_fn=None), flat, name='fully_connected_sequence')
    return ret


class Original_Encoder:
    def __init__(self, sequence, feature_map_count=[3, 96, 128, 256, 256, 256, 256], initializer=None):
        assert (len(feature_map_count) == 7)
        if initializer is None:
            init = tf.random_normal_initializer()
        else:
            init = initializer

        # block 0
        conv0 = conv_sequence(
            sequence, feature_map_count[0], feature_map_count[1], K=7, initializer=init)
        pool0 = max_pool_sequence(conv0)
        relu0 = relu_sequence(pool0)

        # block 1
        conv1 = conv_sequence(
            relu0, feature_map_count[1], feature_map_count[2], initializer=init)
        pool1 = max_pool_sequence(conv1)
        relu1 = relu_sequence(pool1)

        # block 2
        conv2 = conv_sequence(
            relu1, feature_map_count[2], feature_map_count[3], initializer=init)
        pool2 = max_pool_sequence(conv2)
        relu2 = relu_sequence(pool2)

        # block 3
        conv3 = conv_sequence(
            relu2, feature_map_count[3], feature_map_count[4], initializer=init)
        pool3 = max_pool_sequence(conv3)
        relu3 = relu_sequence(pool3)

        # block 4
        conv4 = conv_sequence(
            relu3, feature_map_count[4], feature_map_count[5], initializer=init)
        pool4 = max_pool_sequence(conv4)
        relu4 = relu_sequence(pool4)

        # block 4
        conv5 = conv_sequence(
            relu4, feature_map_count[5], feature_map_count[6], initializer=init)
        pool5 = max_pool_sequence(conv5)
        relu6 = relu_sequence(pool5)

        # final block
        flat = flatten_sequence(relu6)
        self.out_tensor = relu_sequence(flat)


class Original_Encoder_old:
    def __init__(self, prev_layer, feature_map_count=[96, 128, 256, 256, 256, 256]):

        assert (len(feature_map_count) == 6)
        self.out_tensor = prev_layer
        kernel_shape = [3, 3]

        for i in range(7):
            if i < 6:
                with tf.name_scope("conv_block"):
                    kernel_shape = [7, 7] if i is 0 else kernel_shape
                    self.out_tensor = tf.map_fn(lambda a: tf.layers.conv2d(
                        a, filters=feature_map_count[i], padding='SAME', kernel_size=kernel_shape, activation=None), self.out_tensor, name="conv2_map")
                    self.out_tensor = tf.map_fn(
                        lambda a: tf.layers.max_pooling2d(a, 2, 2), self.out_tensor, name="max_pool_map")
                    self.out_tensor = tf.map_fn(
                        tf.nn.relu,  self.out_tensor, name="relu_map")
            elif i == 6:
                self.out_tensor = tf.map_fn(
                    tf.contrib.layers.flatten,  self.out_tensor, name="flatten_map")
                self.out_tensor = tf.map_fn(lambda a: tf.contrib.layers.fully_connected(
                    a, 1024, activation_fn=None), self.out_tensor, name='fully_connected_map')
                self.out_tensor = tf.map_fn(
                    tf.nn.relu,  self.out_tensor, name="relu_map")
