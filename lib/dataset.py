"""deals with data for project"""
import os
import sys
import math
import random
import tarfile

import numpy as np
import pandas as pd
from collections import deque
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


def get_suffeled_batchs(data, label, batch_size):
    assert(len(data) == len(label))
    num_of_batches = math.ceil(len(data)/batch_size)
    perm = np.random.permutation(len(data))
    data_batchs = np.array_split(data[perm], num_of_batches)
    label_batchs = np.array_split(label[perm], num_of_batches)

    return deque(data_batchs), deque(label_batchs)


def read_paths(paths_dir="out/paths.csv"):
    return pd.read_csv(paths_dir, index_col=0).as_matrix()


def get_preprocessed_dataset():
    # get data
    data_all = np.array(sorted(path.construct_path_lists("out", "data_")))
    label_all = np.array(sorted(path.construct_path_lists("out", "labels_")))

    return data_all, label_all


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
