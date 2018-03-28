import os
import re
import json
import sys
import math
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import exposure
from PIL import Image


def vis_montage(im, axis, f_name=None):
    ret_im = exposure.rescale_intensity(montage(im, axis))
    return vis_im(ret_im, f_name)


def vis_im(im, f_name=None):
    fig = plt.figure()
    if f_name is None:
        return plt.imshow(im)
    plt.imsave(f_name, im)
    plt.clf()
    plt.close()


def vis_multichannel(im, f_name=None):
    mulitchannel_montage = montage_multichannel(im)
    return vis_im(mulitchannel_montage, f_name)


def vis_sequence(im, f_name=None):
    sequence_montage = montage_sequence(im)
    return vis_im(sequence_montage, f_name)


def vis_voxel(vox, color=None, f_name=None):

    if color is None:
        color = vox.astype(int)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    color_map = plt.get_cmap('RdYlBu')
    ax.voxels(vox, facecolors=color_map(color), edgecolor='k')

    if f_name is None:
        return fig.show()

    fig.savefig(f_name, bbox_inches='tight')
    fig.clf()
    plt.close()


def vis_softmax(y_hat, f_name=None):
    return vis_voxel(np.argmax(y_hat, axis=-1), y_hat[:, :, :, 1], f_name=f_name)


def montage_multichannel(im):
    return montage(im, -1)


def montage_sequence(im):
    return montage(im, 0)


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
