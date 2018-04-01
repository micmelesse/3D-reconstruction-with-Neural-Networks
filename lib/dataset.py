"""deals with data for project"""
import os
import sys
import math
import random
import tarfile
import numpy as np
import pandas as pd
from collections import deque
from third_party import binvox_rw
from lib import path, utils, render
from sklearn import model_selection
from keras.utils import to_categorical


def load_data(data_samples):
    if data_samples.ndim == 1:
        data_samples = [data_samples]
    ret = []
    for d in data_samples:
        data_row = render.get_render_sequence(d)
        ret.append(data_row)

    return (np.stack(ret) if len(ret) != 1 else ret[0])


def load_label(label_samples):
    if isinstance(label_samples, str):
        label_samples = [label_samples]

    ret = []
    for voxel_path in label_samples:
        with open(voxel_path, 'rb') as f:
            ret.append(binvox_rw.read_as_3d_array(f).data)

    return (np.stack(ret) if len(ret) != 1 else ret[0])


def to_npy(out_dir, arr):
    np.save(out_dir, arr)


def from_npy(npy_path):
    if isinstance(npy_path, str):
        return np.expand_dims(np.load(npy_path), 0)
    ret = []
    for p in npy_path:
        ret.append(np.load(p))
    return np.stack(ret)


def to_npy_data_N_label(paths, N=None):
    if N is None or N <= 0 or N >= len(paths):
        N = len(paths)

    print("convert {} datapoints and labels to npy".format(N))
    for i in range(N):
        model_name = paths[i, 0]
        to_npy('out/{}_x'.format(model_name),
               load_data(paths[i, 1:-1]))
        to_npy('out/{}_y'.format(model_name),
               (to_categorical(load_label(paths[i, -1]))).astype(np.uint8))


def get_suffeled_batchs(data, label, batch_size):
    assert(len(data) == len(label))
    num_of_batches = math.ceil(len(data)/batch_size)
    perm = np.random.permutation(len(data))
    data_batchs = np.array_split(data[perm], num_of_batches)
    label_batchs = np.array_split(label[perm], num_of_batches)

    return deque(data_batchs), deque(label_batchs)


def train_val_test_split(data, label, split=0.1):
    # split into training and test set
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        data, label, test_size=split)  # shuffled
    # split of validation set
    X_train, X_val, y_train, y_val = model_selection.train_test_split(
        X_train, y_train, test_size=split)  # shuffled

    return X_train, y_train, X_val, y_val, X_test, y_test


def read_paths(paths_dir="out/paths.csv"):
    return pd.read_csv(paths_dir, index_col=0).as_matrix()


# get data and labels
def get_preprocessed_dataset():
    data_all = sorted(path.construct_path_lists("out", ["_x.npy"]))
    label_all = sorted(path.construct_path_lists("out", ["_y.npy"]))
    return np.array(data_all), np.array(label_all)


def load_preprocessed_sample():
    data_all = sorted(path.construct_path_lists("out", ["_x.npy"]))
    label_all = sorted(path.construct_path_lists("out", ["_y.npy"]))
    i = np.random.randint(0, len(data_all))
    return np.load(data_all[i]), np.load(label_all[i])


def load_model_testset(epoch_dir):
    model_dir = os.path.dirname(epoch_dir)
    X_test = np.load(
        "{}/X_test.npy".format(model_dir))
    y_test = np.load(
        "{}/y_test.npy".format(model_dir))
    return X_test, y_test


def main():
    example_count = utils.read_params()['TRAIN_PARAMS']['SAMPLE_SIZE']
    if not os.path.isfile("out/paths.csv"):
        path.write_path_csv("data/ShapeNetRendering", "data/ShapeNetVox32")
    to_npy_data_N_label(read_paths(), N=example_count)


if __name__ == '__main__':
    main()
