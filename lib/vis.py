import io
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
from lib import utils


def save_im(im, f_name=None, ndarray=False):
    fig = plt.figure()
    if ndarray:
        fig.set_tight_layout(True)
        fig.canvas.draw()
        ret = np.array(fig.canvas.renderer._renderer)
        fig.clf()
        plt.close()
        return ret

    if f_name is not None:
        plt.imsave(f_name, im)
        plt.clf()
        plt.close()

    return plt.imshow(im)


def voxel(vox, color=None, f_name=None, ndarray=False):
    assert(vox.ndim == 3)

    vox = vox.transpose(2, 0, 1)
    color = color.transpose(2, 0, 1)
    if color is None or len(np.unique(color)) <= 2:
        color = 'red'
    else:
        color_map = plt.get_cmap('coolwarm')
        color = color_map(color)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ret = ax
    ax.voxels(vox, facecolors=color, edgecolor='k')
    ax.view_init(30, 45)

    if ndarray:
        return ret

    if f_name is not None:
        fig.savefig(f_name, bbox_inches='tight')
        fig.clf()
        plt.close()

    return fig.show()


def voxel_binary(y_hat, f_name=None):
    return voxel(np.argmax(y_hat, axis=-1), y_hat[:, :, :, 1], f_name=f_name)


def voxel_ndarray(y_hat):
    return voxel(np.argmax(y_hat, axis=-1), y_hat[:, :, :, 1], ndarray=True)


def label(y, f_name=None):
    return voxel(np.argmax(y, axis=-1), f_name=f_name)


def scaled(im, axis, f_name=None):
    ret_im = exposure.rescale_intensity(montage(im, axis))
    return save_im(ret_im, f_name)


def multichannel(im, f_name=None):
    mulitchannel_montage = flatten_multichannel(im)
    return save_im(mulitchannel_montage, f_name)


def img_sequence(im, f_name=None):
    sequence_montage = flatten_sequence(im)
    return save_im(sequence_montage, f_name)


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
                im = utils.hstack(im, packed_ims[ind])
            else:
                im = utils.hstack(im, np.zeros_like(packed_ims[0]))
        rows.append(im)

    matrix = rows[0]
    for i in range(1, len(rows)):
        matrix = utils.vstack(matrix, rows[i])
    return matrix


def flatten_multichannel(im):
    return montage(im, -1)


def flatten_sequence(im):
    return montage(im, 0)


def create_video(im_list):
    pass


def get_pylab_image(ax):
    im = Image.open(ax.get_array())
    return im
    # im.show()
    # buf.close()


def sample(X, y, yp,  f_name=None):
    X = flatten_sequence(X)
    y = voxel_binary(y)
    yp = voxel_binary(yp)
    n_r = 1
    n_c = 3

    plt.subplot(n_r, n_c, 1)
    plt.imshow(X)

    plt.subplot(n_r, n_c, 2)
    plt.imshow(y)

    plt.subplot(n_r, n_c, 3)
    plt.imshow(yp)

    gcf = plt.gcf()
    gcf.set_size_inches(100, 92)

    if f_name is not None:
        plt.savefig(f_name, bbox_inches='tight')
        plt.clf()
        plt.close()

    return plt.show()
