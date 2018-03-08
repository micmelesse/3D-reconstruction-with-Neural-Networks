import os
import re
import sys
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def vis_im(im, f_name=None):
    fig = plt.figure()
    if f_name is None:
        return plt.imshow(im)

    plt.imsave(f_name, im)
    plt.clf()
    plt.close()


def vis_multichannel(im, f_name=None):
    fig = plt.figure()
    mulitchannel_montage = montage_multichannel(im)
    if f_name is None:
        return plt.imshow(mulitchannel_montage)

    plt.imsave(f_name, mulitchannel_montage)
    plt.clf()
    plt.close()
    return


def vis_sequence(im, f_name=None):
    fig = plt.figure()
    sequence_montage = montage_sequence(im)
    if f_name is None:
        return plt.imshow(sequence_montage)

    plt.imsave(f_name, sequence_montage)
    plt.clf()
    plt.close()
    return


def vis_voxel(vox, f_name=None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(vox, edgecolor='k')
    ax.view_init(30, 30)
    if f_name is None:
        return plt.show()

    plt.savefig(f_name, bbox_inches='tight')
    plt.clf()
    plt.close()


def montage_sequence(im_seqeunce):
    return montage(im_seqeunce, 0)


def montage_multichannel(im_multichannel):
    return montage(im_multichannel, -1)


def montage(packed_ims, axis):
    """display as an Image the contents of packed_ims in a square gird along an aribitray axis"""
    if packed_ims.ndim == 2:
        return packed_ims

    # bring axis to the front
    packed_ims = np.rollaxis(packed_ims, axis)

    N = len(packed_ims)
    n_tile = math.ceil(math.sqrt(N))
    rows = []
    for i in range(n_tile):
        im = packed_ims[i * n_tile]
        for j in range(1, n_tile):
            ind = i * n_tile + j
            if ind < N:
                im = hstack(im, packed_ims[ind])
            else:
                im = hstack(im, np.zeros_like(packed_ims[0]))
        rows.append(im)

    matrix = rows[0]
    for i in range(1, len(rows)):
        matrix = vstack(matrix, rows[i])
    return matrix


def hstack(a, b):
    return np.hstack((a, b))


def vstack(a, b):
    return np.vstack((a, b))


def check_dir():
    TRAIN_DIRS = ["out", "params", "data", "aws"]
    for d in TRAIN_DIRS:
        if not os.path.isdir(d):
            os.makedirs(d)


def read_param(param_line):
    regex = "^.*=(.*)$"
    return re.findall(regex, param_line)[0]


def get_params_from_disk():
    f = None

    try:
        f = open("params/train.params")
    except:
        pass

    try:
        learn_rate = float(read_param(f.readline()))
    except:
        learn_rate = None

    try:
        batch_size = int(read_param(f.readline()))
    except:
        batch_size = None

    try:
        epoch = int(read_param(f.readline()))
    except:
        epoch = None

    if f:
        f.close()

    return learn_rate, batch_size, epoch
