
import os
import re
import glob
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
from natsort import natsorted
from filecmp import dircmp
from lib import dataset, encoder, recurrent_module, decoder, loss, vis, utils


# inspired by numpy move axis function
# def tf_move_axis(X, src, dst):
#     ndim = len(X.get_shape())
#     order = [for i in range(ndim)]


def is_epoch_dir(epoch_dir):
    return "epoch_" in epoch_dir


def get_model_info(model_dir):
    if is_epoch_dir(model_dir):
        model_dir = os.path.dirname(model_dir)

    model_info = {}
    i = 0
    while os.path.exists(os.path.join(model_dir, "epoch_{}".format(i))):
        i += 1
    model_info["EPOCH_COUNT"] = i
    model_info["EPOCH_INDEX"] = i-1
    return model_info


def get_model_predictions(obj_id, model_dir):
    model_info = get_model_info(model_dir)
    x, y = dataset.load_obj_id(grep_obj_id(obj_id))
    for i in range(model_info["EPOCH_COUNT"]):
        net = network.Network_restored("{}/epoch_{}".format(model_dir, i))
        yp = net.predict(x)


def filter_files(regex):
    return natsorted(glob.glob(regex, recursive=True))


def list_folders(path="."):
    folder_list = sorted(next(os.walk(path))[1])
    ret = []
    for f in folder_list:
        ret.append(path+"/"+f)
    return ret



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
    s = os.path.basename(s)
    regex = "(.*_.*)_(x|y|yp|p|sm).(png|npy)$"
    return re.search(regex, s).group(1)


def grep_stepcount(s):
    s = os.path.basename(s)
    regex = "(.*)_(.*_.*)_(x|y|yp|p|sm).(png|npy)$"
    return re.search(regex, s).group(1)


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
