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


def save_im(im, f_name=None):
    fig = plt.figure()
    if f_name is None:
        return plt.imshow(im)
    plt.imsave(f_name, im)
    plt.clf()
    plt.close()


def voxel(vox, color=None, f_name=None):

    if color is None:
        color = 'red'
    else:
        color_map = plt.get_cmap('coolwarm')
        color = color_map(color)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(vox, facecolors=color, edgecolor='k')

    if f_name is None:
        return fig.show()

    fig.savefig(f_name, bbox_inches='tight')
    fig.clf()
    plt.close()


def label(y, f_name=None):
    return voxel(np.argmax(y, axis=-1), f_name=f_name)


def softmax(y_hat, f_name=None):
    return voxel(np.argmax(y_hat, axis=-1), y_hat[:, :, :, 1], f_name=f_name)


def scaled(im, axis, f_name=None):
    ret_im = exposure.rescale_intensity(utils.montage(im, axis))
    return save_im(ret_im, f_name)


def multichannel(im, f_name=None):
    mulitchannel_montage = utils.montage_multichannel(im)
    return save_im(mulitchannel_montage, f_name)


def sequence(im, f_name=None):
    sequence_montage = utils.montage_sequence(im)
    return save_im(sequence_montage, f_name)
