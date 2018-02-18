import os
import re
import sys
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# def plot_features(im):
#     print im.shape
def read_param(param_line):
    regex = "^.*=(.*)$"
    return re.findall(regex, param_line)[0]


def get_params_from_disk():
    with open("config/train.params") as f:
        learn_rate = float(read_param(f.readline()))
        batch_size = int(read_param(f.readline()))
        epoch = int(read_param(f.readline()))
    return learn_rate, batch_size, epoch


def imshow_sequence(im):
    return plt.imshow(flatten_sequence(im))


def imsave_sequence(im, f_name="test.png"):
    plt.imsave(f_name, (flatten_sequence(im)))
    plt.clf()
    plt.close()
    return


def imshow_multichannel(im):
    return plt.imshow(flatten_multichannel_image(im))


def imsave_multichannel(im, f_name="test.png"):
    plt.imsave(f_name, (flatten_multichannel_image(im)))
    plt.clf()
    plt.close()
    return

# vis voxels


def imshow_voxel(vox):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(vox, edgecolor='k')
    ax.view_init(30, 30)
    return plt.show()


def imsave_voxel(vox, f_name="test.png"):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(vox, edgecolor='k')
    ax.view_init(30, 30)
    plt.savefig(f_name, bbox_inches='tight')
    plt.clf()
    plt.close()


def flatten_multichannel_image(im):
    # print(im.shape)

    if im.ndim == 2:
        return im

    n_channels = im.shape[-1]
    n_tile = math.ceil(math.sqrt(n_channels))
    rows = []
    for i in range(n_tile):
        c_first = i * n_tile
        if c_first < n_channels:
            a = im[:, :, c_first]
            for j in range(1, n_tile):
                c_last = c_first + j
                if c_last < n_channels:
                    b = im[:, :, c_last]
                    a = hstack(a, b)
                else:
                    b = np.zeros([im.shape[0], im.shape[1]],)
                    a = hstack(a, b)
            rows.append(a)

    n = len(rows)
    a = rows[0]
    for i in range(1, n):
        b = rows[i]
        a = np.vstack((a, b))
    return a


def flatten_sequence(im_sequence):

    a = flatten_multichannel_image(im_sequence[0])
    for b in im_sequence[1:]:
        a = hstack(a, flatten_multichannel_image(b))
    return a


def hstack(a, b):
    return np.hstack((a, b))


def vstack(a, b):
    return np.vstack((a, b))


def r2n2_unpool3D(value, name='unpool3D'):
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1: -1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat([out, tf.zeros_like(out)], i)
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out


def r2n2_matmul(a, b):
    # print(a.shape, b.shape)
    ret = tf.expand_dims(a, axis=-2)
    # print(ret.shape, b.shape)
    ret = tf.matmul(ret, b)
    # print(ret.shape)
    ret = tf.squeeze(ret, axis=-2)
    # print(ret.shape)
    return ret


def r2n2_linear(x, W, U, h, b):
    # print(x.shape, W.shape, U.shape, h.shape, b.shape)
    Wx = tf.map_fn(lambda a: r2n2_matmul(a, W), x)
    Uh = tf.nn.conv3d(h, U, strides=[1, 1, 1, 1, 1], padding="SAME")
    # print(Wx.shape, Uh.shape, b.shape)
    return Wx + Uh + b


def r2n2_stack(x, N=4):
    return tf.transpose(tf.stack([tf.stack([tf.stack([x] * N)] * N)] * N), [3, 0, 1, 2, 4])


def to_npy(rows):
    if isinstance(rows, str):
        return np.expand_dims(np.load(rows), 0)
    ret = []
    for r in rows:
        ret.append(np.load(r))
    return np.stack(ret)


def get_batchs(data_all, label_all, batch_size):

    N = len(data_all)
    num_of_batches = math.ceil(N/batch_size)
    assert(N == len(label_all))
    perm = np.random.permutation(N)
    data_all = data_all[perm]
    label_all = label_all[perm]
    data_batchs = np.array_split(data_all, num_of_batches)
    label_batchs = np.array_split(label_all, num_of_batches)

    return data_batchs, label_batchs


def check_dir():
    TRAIN_DIRS = ["out", "config", "data", "aws"]
    for d in TRAIN_DIRS:
        if not os.path.isdir(d):
            os.makedirs(d)
