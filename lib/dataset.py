"""deals with data for project"""
import os
import sys
import math
import random
import tarfile
import numpy as np
import pandas as pd
from lib import path, utils, render
from third_party import binvox_rw


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
        to_npy('out/data_{:06d}'.format(i),
               load_data(paths[i, 0:-2]))
        to_npy('out/labels_{:06d}'.format(i), load_label(paths[i, -2]))


def get_batchs(data_all, label_all, batch_size):

    N = len(data_all)
    num_of_batches = math.ceil(N/batch_size)
    assert(N == len(label_all))
    perm = np.random.permutation(N)
    data_all = data_all[perm]
    label_all = label_all[perm]
    data_batchs = np.array_split(data_all, num_of_batches)
    label_batchs = np.array_split(label_all, num_of_batches)

    return data_batchs, label_batchs


def read_paths(paths_dir="out/paths.csv"):
    return pd.read_csv(paths_dir, index_col=0).as_matrix()


class Dataset:  # deals with data
    def __init__(self):
        self.paths = read_paths()
        np.random.shuffle(self.paths)
        self.N = self.paths.shape[0]
        self.split_index = math.ceil(self.N * 0.8)
        self.train_index = 0
        self.test_index = self.split_index
        self.batch_size = 36

    def next_train_batch(self, batch_size=None):
        paths_ls = self.next_train_batch_paths(batch_size)
        if paths_ls is not None:
            data_label_tuple = (load_data_matrix(
                paths_ls[:, 0:-2]), load_labels(paths_ls[:, -2]))
        if data_label_tuple[0] is None:
            return None, None
        return data_label_tuple

    def next_test_batch(self, batch_size=None):
        paths_ls = self.next_test_batch_paths(batch_size)
        if paths_ls is not None:
            data_label_tuple = (load_data_matrix(
                paths_ls[:, 0:-2]), load_labels(paths_ls[:, -2]))
        if data_label_tuple[0] is None:
            return None, None
        return data_label_tuple

    def next_train_batch_paths(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        prev_index = self.train_index
        self.train_index += batch_size

        if self.train_index >= self.split_index:
            self.train_index = 0
            return None
        else:
            return self.paths[prev_index:self.train_index]

    def next_test_batch_paths(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        prev_index = self.test_index
        self.test_index += batch_size

        if self.test_index >= self.N:
            self.test_index = self.split_index
            return None
        else:
            return self.paths[prev_index:self.test_index]

    def reset(self):
        np.random.shuffle(self.paths)
        self.train_index = 0
        self.test_index = self.split_index


def main():
    f = None
    try:
        f = open("params/dataset.params")
    except:
        pass

    try:
        example_count = int(utils.read_param(f.readline()))
    except:
        example_count = None

    if f:
        f.close()

    if not os.path.isfile("out/paths.csv"):
        path.write_path_csv("data/ShapeNetRendering", "data/ShapeNetVox32")

    to_npy_data_N_label(read_paths(), N=example_count)
