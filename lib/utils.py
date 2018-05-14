
import os
import re
import glob
import trimesh
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
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from lib import dataset, network, encoder, recurrent_module, decoder, loss, vis, utils


# inspired by numpy move axis function
# def tf_move_axis(X, src, dst):
#     ndim = len(X.get_shape())
#     order = [for i in range(ndim)]


def to_npy(out_dir, arr):
    np.save(out_dir, arr)


def load_npy(npy_path):
    if isinstance(npy_path, str):
        return np.expand_dims(np.load(npy_path), 0)
    ret = []
    for p in npy_path:
        ret.append(np.load(p))
    return np.stack(ret)


def is_epoch_dir(epoch_dir):
    return "epoch_" in epoch_dir


def get_latest_epoch_index(model_dir):
    if is_epoch_dir(model_dir):
        model_dir = os.path.dirname(model_dir)
    i = 0
    while os.path.exists(os.path.join(model_dir, "epoch_{}".format(i))):
        i += 1
    return i-1


def get_latest_epoch(model_dir):
    return model_dir + "/epoch_{}".format(get_latest_epoch_index(model_dir))


def get_latest_loss(model_dir, loss_type="train"):
    epoch = get_latest_epoch(model_dir)
    epoch_prev = model_dir + \
        "/epoch_{}".format(get_latest_epoch_index(model_dir)-1)

    try:
        return np.load(epoch+"/{}_loss.npy".format(loss_type))
    except:
        return np.load(epoch_prev+"/{}_loss.npy".format(loss_type))


def get_model_params(model_dir):
    json_list = dataset.construct_file_path_list_from_dir(model_dir, ".json")
    if json_list:
        return read_params(json_list[0])
    return {}


def get_model_predictions(obj_id, model_dir):
    epoch_count = get_latest_epoch_index(model_dir)+1
    x, y = dataset.load_obj_id(grep_obj_id(obj_id))
    for i in range(epoch_count):
        net = network.Network_restored("{}/epoch_{}".format(model_dir, i))
        yp = net.predict(x)


def get_model_dataset_split(model_dir):
    try:
        X_train = np.load("{}/X_train.npy".format(model_dir))
    except:
        X_train = None

    try:
        X_train = np.load("{}/X_train.npy".format(model_dir))
    except:
        X_train = None

    try:
        y_train = np.load("{}/y_train.npy".format(model_dir))
    except:
        y_train = None

    try:
        X_val = np.load("{}/X_val.npy".format(model_dir))
    except:
        X_val = None

    try:
        y_val = np.load("{}/y_val.npy".format(model_dir))
    except:
        y_val = None

    try:
        X_test = np.load("{}/X_test.npy".format(model_dir))
    except:
        X_test = None
    try:
        y_test = np.load("{}/y_test.npy".format(model_dir))
    except:
        y_test = None

    return X_train, X_val, X_test, y_train, y_val, y_test


def filter_files(regex):
    return natsorted(glob.glob(regex, recursive=True))


def list_folders(path="."):
    folder_list = sorted(next(os.walk(path))[1])
    ret = []
    for f in folder_list:
        ret.append(path+"/"+f)
    return ret


def check_params_json(param_json="params.json"):
    if not os.path.exists(param_json):
        param_data = {}
        with open(param_json, 'w') as param_file:
            json.dump(param_data, param_file)


def read_params(params_json="params.json"):
    check_params_json(params_json)
    return json.loads(open(params_json).read())


def fix_nparray(path):
    arr = np.load(path)
    N = len(arr)
    l = arr[0]
    for i in range(1, N):
        l += arr[i]
    np.save(path, np.array(l))


def replace_with_flat(path):
    arr = np.load(path)
    np.save(path, arr.flatten())


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


def grep_timestamp(s):
    regex = ".*model_(.*)_(.*)"
    ret = re.search(regex, s)
    return ret.group(1), ret.group(2)


def make_dir(file_dir):
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)


def make_prev_dirs(file_dir):
    file_dir = os.path.dirname(file_dir)
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


def get_summary_as_array(model_dir, scalar="loss", run="train"):
    name = "/{}_{}.npy".format(run, scalar)
    if os.path.exists(model_dir+name):
        return np.load(model_dir+name)

    event_file_path = glob.glob(model_dir+"/{}/event*".format(run))[0]
    event_acc = EventAccumulator(event_file_path)
    event_acc.Reload()
    ret = np.stack(
        [np.asarray([s.step, s.value])
         for s in event_acc.Scalars(scalar)])
    np.save(model_dir+name, ret)

    return ret
