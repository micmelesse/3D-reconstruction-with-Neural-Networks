import tensorflow as tf
from lib import utils


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


def conv_vox(vox, fv_count_in, fv_count_out, K=3, S=[1, 1, 1, 1, 1], D=[1, 1, 1, 1, 1], initializer=None, P="SAME"):
    with tf.name_scope("conv_vox"):
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        kernel = tf.Variable(
            init([K, K, K, fv_count_in, fv_count_out]), name="kernel")
        bias = tf.Variable(init([fv_count_out]), name="bias")
        ret = tf.nn.bias_add(tf.nn.conv3d(
            vox, kernel, S, padding=P, dilations=D, name="conv3d"), bias)

        params = utils.read_params()
        if params["VIS"]["HISTOGRAM"]:
            tf.summary.histogram("kernel", kernel)
            tf.summary.histogram("bias", bias)

    return ret


def simple_decoder_block(vox, fv_count_in, fv_count_out, K=3, D=[1, 1, 1, 1, 1], initializer=None, unpool=False):
    with tf.name_scope("simple_decoder_block"):
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        conv = conv_vox(vox, fv_count_in, fv_count_out,
                        K=K,  D=D, initializer=init)
        if unpool:
            return relu_vox(unpool_vox(conv))

    return relu_vox(conv)


def residual_decoder_block(vox, fv_count_in, fv_count_out, K_1=3, K_2=3, D=[1, 1, 1, 1, 1], initializer=None, unpool=False):
    with tf.name_scope("residual_decoder_block"):
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        conv1 = conv_vox(vox, fv_count_in,
                         fv_count_out, K=K_1, D=D, initializer=init)
        relu1 = relu_vox(conv1)
        conv2 = conv_vox(relu1, fv_count_out,
                         fv_count_out, K=K_2, D=D, initializer=init)
        relu2 = relu_vox(conv2)
        out = vox + relu2

        if unpool:
            return relu_vox(unpool_vox(out))

    return relu_vox(out)


class Residual_Decoder:
    def __init__(self, hidden_state, feature_vox_count=[128, 128, 128, 64, 32, 2], initializer=None):
        with tf.name_scope("Simple_Decoder"):
            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            N = len(feature_vox_count)
            cur_tensor = unpool_vox(hidden_state)
            cur_tensor = residual_decoder_block(
                cur_tensor, 128, feature_vox_count[0], initializer=init)
            for i in range(1, N-1):
                unpool = True if i < 3 else False
                cur_tensor = residual_decoder_block(
                    cur_tensor, feature_vox_count[i-1], feature_vox_count[i], initializer=init, unpool=unpool)

            self.out_tensor = conv_vox(
                cur_tensor, feature_vox_count[-2], feature_vox_count[-1], initializer=init)


class Simple_Decoder:
    def __init__(self, hidden_state, feature_vox_count=[128, 128, 128, 64, 32, 2], initializer=None):
        with tf.name_scope("Simple_Decoder"):
            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            N = len(feature_vox_count)
            cur_tensor = unpool_vox(hidden_state)
            cur_tensor = simple_decoder_block(
                cur_tensor, 128, feature_vox_count[0], initializer=init)
            for i in range(1, N-1):
                unpool = True if i < 3 else False
                cur_tensor = simple_decoder_block(
                    cur_tensor, feature_vox_count[i-1], feature_vox_count[i], initializer=init, unpool=unpool)

            self.out_tensor = conv_vox(
                cur_tensor, feature_vox_count[-2], feature_vox_count[-1], initializer=init)


class Dilated_Decoder:
    def __init__(self, hidden_state, feature_vox_count=[128, 128, 128, 64, 32, 2], initializer=None):
        with tf.name_scope("Simple_Decoder"):
            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            N = len(feature_vox_count)
            cur_tensor = unpool_vox(hidden_state)
            cur_tensor = simple_decoder_block(
                cur_tensor, 128, feature_vox_count[0], initializer=init)
            for i in range(1, N-1):
                unpool = True if i < 3 else False
                cur_tensor = simple_decoder_block(
                    cur_tensor, feature_vox_count[i-1], feature_vox_count[i], D=[1, 2, 2, 2, 1], initializer=init, unpool=unpool)

            self.out_tensor = conv_vox(
                cur_tensor, feature_vox_count[-2], feature_vox_count[-1], initializer=init)
