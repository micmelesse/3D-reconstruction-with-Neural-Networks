import os
import re
import json
import sys
import math
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import exposure
from PIL import Image

from filecmp import dircmp


def list_folders(path="."):
    folder_list = sorted(next(os.walk(path))[1])
    ret = []
    for f in folder_list:
        ret.append(path+"/"+f)
    return ret


def prep_dir():
    TRAIN_DIRS = ["out", "data", "aws", "results"]
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


def fix_nparray(path):
    arr = np.load(path)
    l = []
    for i in arr:
        l += i
    np.save(path, np.array(l))


def grep_params(s):
    regex = "^.*=(.*)$"
    return re.findall(regex, s)[0]


def grep_epoch_name(epoch_dir):
    return re.search(".*(epoch_.*).*", epoch_dir).group(1)


def grep_learning_rate(s):
    regex = "^.*L:(.*?)_.*$"
    return float(re.findall(regex, s)[0])


def grep_batch_size(s):
    regex = "^.*B:(.*?)_.*$"
    return float(re.findall(regex, s)[0])


def grep_epoch_count(s):
    regex = "^.*E:(.*?)_.*$"
    return float(re.findall(regex, s)[0])


def grep_obj_id(s):
    regex = "^(.*)_(.*_.*)_(x|y|yp|p|sm).(png|npy)$"
    return re.search(regex, os.path.basename(s)).group(2)


def grep_stepcount(s):
    regex = "^(.*)_(.*_.*)_(x|y|yp|p|sm).(png|npy)$"
    return re.search(regex, os.path.basename(s)).group(1)


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





