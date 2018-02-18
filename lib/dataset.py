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


class ShapeNet:
    def __init__(self):
        self.paths = pd.read_csv("out/paths.csv", index_col=0).as_matrix()
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


def load_data(data_samples):
    if data_samples.ndim == 1:
        data_samples = [data_samples]
    ret = []
    for d in data_samples:
        data_row = render.get_render_sequence(d)
        ret.append(data_row)

    return (np.stack(ret) if len(ret) != 1 else ret[0])


def load_labels(label_samples):
    if isinstance(label_samples, str):
        label_samples = [label_samples]

    ret = []
    for voxel_path in label_samples:
        with open(voxel_path, 'rb') as f:
            ret.append(binvox_rw.read_as_3d_array(f).data)

    return (np.stack(ret) if len(ret) != 1 else ret[0])


def save_data_to_npy(paths, N=None):
    if N is None or N <= 0 or N >= len(paths):
        N = len(paths)

    print("data and labels for {} examples".format(N))
    for i in range(N):
        np.save('out/data_{:06d}'.format(i),
                load_data(paths[i, 0:-2]))
        np.save('out/labels_{:06d}'.format(i), load_labels(paths[i, -2]))


def main():
    with open("config/dataset.params") as f:
        example_count = int(utils.read_param(f.readline()))

    if not os.path.isfile("out/paths.csv"):
        path.write_path_csv("data/ShapeNetRendering", "data/ShapeNetVox32")

    shapenet = ShapeNet()
    save_data_to_npy(shapenet.paths, N=example_count)
