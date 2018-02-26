import os
import re
import sys
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


def read_param(param_line):
    regex = "^.*=(.*)$"
    return re.findall(regex, param_line)[0]


def get_params_from_disk():
    with open("params/train.params") as f:
        try:
            learn_rate = float(read_param(f.readline()))
            batch_size = int(read_param(f.readline()))
            epoch = int(read_param(f.readline()))
        except:
            learn_rate = None
            batch_size = None
            epoch = None

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
    TRAIN_DIRS = ["out", "params", "data", "aws"]
    for d in TRAIN_DIRS:
        if not os.path.isdir(d):
            os.makedirs(d)

    def get_encoder_state(self, x, y, state_dir="./out/state/"):
        if not os.path.isdir(state_dir):
            os.makedirs(state_dir)
        fd = {self.X: x, self.Y: y}
        states_all = []

        n_encoder = len(self.encoder_outputs)
        for l in range(n_encoder):
            state = self.encoder_outputs[l].eval(fd)
            states_all.append(state)

        return states_all

    def get_hidden_state(self, x, y, state_dir="./out/state/"):
        if not os.path.isdir(state_dir):
            os.makedirs(state_dir)
        fd = {self.X: x, self.Y: y}
        states_all = []

        n_hidden = len(self.hidden_state_list)
        for l in range(n_hidden):
            state = self.hidden_state_list[l].eval(fd)
            states_all.append(state)

        return states_all

    def get_decoder_state(self, x, y, state_dir="./out/state/"):
        if not os.path.isdir(state_dir):
            os.makedirs(state_dir)
        fd = {self.X: x, self.Y: y}
        states_all = []

        n_decoder = len(self.decoder_outputs)
        for l in range(n_decoder):
            state = self.decoder_outputs[l].eval(fd)
            states_all.append(state)

        states_all.append(self.softmax.eval(fd))
        return states_all
