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


def montage_multichannel(im):
    return montage(im, -1)


def montage_sequence(im):
    return montage(im, 0)


def prep_dir():
    TRAIN_DIRS = ["out", "data", "aws"]
    for d in TRAIN_DIRS:
        make_dir(d)

    param_data = {
        "TRAIN_PARAMS": {
        },
        "AWS_PARAMS": {
        },
        "DIRS": {
        }
    }
    param_name = "params.json"

    if not os.path.exists(param_name):
        with open(param_name, 'w') as param_file:
            json.dump(param_data, param_file)


def read_params(json_dir="params.json"):
    return json.loads(open(json_dir).read())


def grep_epoch_name(epoch_dir):
    return re.search(".*(epoch_.*).*", epoch_dir).group(1)


def grep_params(param_line):
    regex = "^.*=(.*)$"
    return re.findall(regex, param_line)[0]


def make_dir(file_dir):
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)


def clean_dir(file_dir):
    if os.path.isdir(file_dir):
        shutil.rmtree(file_dir)
        os.makedirs(file_dir)


def get_file_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def hstack(a, b):
    return np.hstack((a, b))


def vstack(a, b):
    return np.vstack((a, b))
