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
from lib import utils, dataset, network
from moviepy.video.io.bindings import mplfig_to_npimage


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
        utils.make_prev_dirs(f_name)
        plt.imsave(f_name, im)
        plt.clf()
        plt.close()

    return plt.imshow(im)


def voxel(vox, color=None, f_name=None, npimage=False):
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
    ax.voxels(vox, facecolors=color, edgecolor='k')
    ax.view_init(30, 45)

    if npimage:
        return mplfig_to_npimage(fig)

    if f_name is not None:
        utils.make_prev_dirs(f_name)
        fig.savefig(f_name, bbox_inches='tight')
        fig.clf()
        plt.close()
        return

    return fig.show()


def voxel_binary(y_hat, f_name=None):
    vox = np.argmax(y_hat, axis=-1)
    color = y_hat[:, :, :, 1]
    return voxel(vox, color, f_name=f_name)


def voxel_npimage(y_hat):
    vox = np.argmax(y_hat, axis=-1)
    color = y_hat[:, :, :, 1]
    return voxel(vox, color, npimage=True)


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


def get_pylab_image(ax):
    im = Image.open(ax.get_array())
    return im
    # im.show()
    # buf.close()


def sample(X, y, yp, f_name=None):

    ax1 = plt.subplot(223)
    ax1.imshow(flatten_sequence(X))

    ax2 = plt.subplot(221, projection='3d')
    vox = (np.argmax(y, axis=-1)).transpose(2, 0, 1)
    color = (plt.get_cmap('coolwarm'))((y[:, :, :, 1]).transpose(2, 0, 1))
    ax2.voxels(vox, facecolors=color, edgecolor='k')
    ax2.view_init(30, 45)

    ax3 = plt.subplot(222, projection='3d')
    vox = (np.argmax(yp, axis=-1)).transpose(2, 0, 1)
    color = (plt.get_cmap('coolwarm'))((yp[:, :, :, 1]).transpose(2, 0, 1))
    ax3.voxels(vox, facecolors=color, edgecolor='k')
    ax3.view_init(30, 45)

    if f_name is not None:
        plt.savefig(f_name)
        plt.clf()
        plt.close()
        return

    return plt.show()


def create_video(obj_id="02691156_131db4a650873babad3ab188d086d4db"):
    params = utils.read_params()
    out_dir = params["DIRS"]["OUTPUT"]
    model_dir = params["SESSIONS"]["LONGEST"]
    model_info = utils.get_model_info(model_dir)
    epoch_count = model_info["EPOCH_COUNT"]

    x, _ = dataset.load_obj_id(obj_id)
    for i in range(epoch_count):
        net = network.Network_restored("{}/epoch_{}".format(model_dir, i))
        yp = net.predict(x)
        voxel_binary(yp[0], f_name="{}/{}/frame_{}".format(out_dir, obj_id, i))
