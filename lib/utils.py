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


def hstack(a, b):
    return np.hstack((a, b))


def vstack(a, b):
    return np.vstack((a, b))


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
