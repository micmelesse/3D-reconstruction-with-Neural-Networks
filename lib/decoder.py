import tensorflow as tf
from lib import utils


def conv_vox(vox, in_featurevoxel_count, out_featurevoxel_count, K=3, S=[1, 1, 1, 1, 1], D=[1, 1, 1, 1, 1], initializer=None, P="SAME"):
    with tf.name_scope("conv_vox"):
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        kernel = tf.Variable(
            init([K, K, K, in_featurevoxel_count, out_featurevoxel_count]), name="kernel")
        bias = tf.Variable(init([out_featurevoxel_count]), name="bias")
        ret = tf.nn.bias_add(tf.nn.conv3d(
            vox, kernel, S, padding=P, dilations=D, name="conv3d"), bias)
        tf.add_to_collection("feature_voxels", ret)

        # visualization code
        params = utils.read_params()
        image_count = params["VIS"]["IMAGE_COUNT"]
        if params["VIS"]["KERNELS"]:
            kern_1 = tf.concat(tf.unstack(kernel, axis=-1), axis=-1)
            kern_2 = tf.transpose(kern_1, [3, 0, 1, 2])
            kern_3 = tf.expand_dims(kern_2, -1)
            kern_4 = tf.concat(tf.unstack(kern_3, axis=1), axis=1)
            tf.summary.image("3d kernel", kern_4, max_outputs=image_count)

        if params["VIS"]["VOXEL_SLICES"]:
            vox_slice_1 = tf.unstack(ret, axis=4)[1]
            vox_slice_2 = tf.split(vox_slice_1, 4, axis=3)
            vox_slice_3 = tf.concat(vox_slice_2, axis=1)
            vox_slice_4 = tf.concat(tf.unstack(vox_slice_3, axis=-1), axis=2)
            vox_slice_5 = tf.expand_dims(vox_slice_4, -1)
            tf.summary.image("vox_slices", vox_slice_5,
                             max_outputs=image_count)

        if params["VIS"]["FEATURE_VOXELS"]:
            tf.summary.tensor_summary("feature_voxels", ret[0, :, :, :, 0])

        if params["VIS"]["HISTOGRAMS"]:
            tf.summary.histogram("kernel", kernel)
            tf.summary.histogram("bias", bias)

        if params["VIS"]["SHAPES"]:
            print(ret.shape)

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


def relu_vox(vox):
    with tf.name_scope("relu_vox"):
        ret = tf.nn.relu(vox, name="relu")
    return ret


def block_simple_decoder(vox, in_featurevoxel_count, out_featurevoxel_count, K=3, D=[1, 1, 1, 1, 1], initializer=None, unpool=False):
    with tf.name_scope("block_simple_decoder"):
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        conv = conv_vox(vox, in_featurevoxel_count, out_featurevoxel_count,
                        K=K,  D=D, initializer=init)
        if unpool:
            out = relu_vox(unpool_vox(conv))
        else:
            out = relu_vox(conv)

    return out


def block_residual_decoder(vox, in_featurevoxel_count, out_featurevoxel_count, K_1=3, K_2=3, K_3=1, D=[1, 1, 1, 1, 1], initializer=None, unpool=False):
    with tf.name_scope("block_residual_decoder"):
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        out = vox
        if K_1 != 0:
            conv1 = conv_vox(out, in_featurevoxel_count,
                             out_featurevoxel_count, K=K_1, D=D, initializer=init)
            relu1 = relu_vox(conv1)
            out = relu1

        if K_2 != 0:
            conv2 = conv_vox(out, out_featurevoxel_count,
                             out_featurevoxel_count, K=K_2, D=D, initializer=init)
            relu2 = relu_vox(conv2)
            out = relu2

        if K_3 != 0:
            conv3 = conv_vox(out, out_featurevoxel_count,
                             out_featurevoxel_count, K=K_3, D=D, initializer=init)
            out = conv3 + relu2

        if unpool:
            unpool = unpool_vox(out)
            out = unpool

    return out


class Residual_Decoder:
    def __init__(self, hidden_state, feature_vox_count=[128, 128, 128, 64, 32, 2], initializer=None):
        with tf.name_scope("Residual_Decoder"):
            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            N = len(feature_vox_count)
            hidden_shape = hidden_state.get_shape().as_list()
            cur_tensor = unpool_vox(hidden_state)
            cur_tensor = block_residual_decoder(
                cur_tensor, hidden_shape[-1], feature_vox_count[0], initializer=init)
            for i in range(1, N-1):
                unpool = True if i <= 2 else False
                cur_tensor = block_residual_decoder(
                    cur_tensor, feature_vox_count[i-1], feature_vox_count[i], initializer=init, unpool=unpool)

            self.out_tensor = conv_vox(
                cur_tensor, feature_vox_count[-2], feature_vox_count[-1], initializer=init)


class Dilated_Decoder:
    def __init__(self, hidden_state, feature_vox_count=[128, 128, 128, 64, 32, 2], initializer=None):
        with tf.name_scope("Dilated_Decoder"):
            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            N = len(feature_vox_count)
            hidden_shape = hidden_state.get_shape().as_list()
            cur_tensor = unpool_vox(hidden_state)
            cur_tensor = block_simple_decoder(
                cur_tensor, hidden_shape[-1], feature_vox_count[0], initializer=init)
            for i in range(1, N-1):
                unpool = True if i <= 2 else False
                cur_tensor = block_simple_decoder(
                    cur_tensor, feature_vox_count[i-1], feature_vox_count[i], D=[1, 2, 2, 2, 1], initializer=init, unpool=unpool)

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
            hidden_shape = hidden_state.get_shape().as_list()
            cur_tensor = unpool_vox(hidden_state)
            cur_tensor = block_simple_decoder(
                cur_tensor, hidden_shape[-1], feature_vox_count[0], initializer=init)
            for i in range(1, N-1):
                unpool = True if i <= 2 else False
                cur_tensor = block_simple_decoder(
                    cur_tensor, feature_vox_count[i-1], feature_vox_count[i], initializer=init, unpool=unpool)

            self.out_tensor = conv_vox(
                cur_tensor, feature_vox_count[-2], feature_vox_count[-1], initializer=init)
