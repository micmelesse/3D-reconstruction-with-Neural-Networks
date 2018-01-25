"""deals with data for project"""
import os
import sys
import random
import tarfile
import math
import pandas as pd
import numpy as np
from filecmp import dircmp
import params as params
import binvox_rw
import render


def save_data_to_npy(paths, N=None):
    # if N is None:
    #     N=len(paths)
    # for i in range(24):
    #     print("save column_{} for {} examples".format(i,N))
    #     column = load_data_matrix(paths[0:N, i])
    #     np.save('column_{}'.format(i), column)

    print("save labels for {} examples".format(N))
    all_labels = load_labels((paths[0:N, -2]))
    np.save('all_labels', all_labels)
    print("save data for {} examples".format(N))
    all_data = load_data_matrix((paths[0:N, 0:-2]))
    np.save('all_data', all_data)


def main():
    with open("config/dataset.params") as f:
        example_count = int(params.read_param(f.readline()))

    shapenet = ShapeNet()
    save_data_to_npy(shapenet.paths, N=example_count)


class ShapeNet:
    def __init__(self):
        self.paths = pd.read_csv("./out/paths.csv", index_col=0).as_matrix()
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


def load_data_matrix(data_columns):
    if isinstance(data_columns, np.ndarray):
        data_columns = data_columns.tolist()

    mat = []
    for c in data_columns:
        mat.append(load_dataset_row(c))
    return np.stack(mat)


def load_dataset_row(data_row):
    if isinstance(data_row, np.ndarray):
        data_row = data_row.tolist()

    return render.fetch_renders_from_disk(data_row)


def load_labels(label_column):
    if isinstance(label_column, np.ndarray):
        label_column = label_column.tolist()

    voxel_list = []
    for voxel_path in label_column:
        with open("./data/" + voxel_path, 'rb') as f:
            voxel_list.append(
                (binvox_rw.read_as_3d_array(f)).data.astype(float))

    return np.stack(voxel_list)


def write_path_csv(data_dir, label_dir):

    common_paths = []
    for dir_top, subdir_cmps in dircmp(data_dir, label_dir).subdirs.items():
        for dir_bot in subdir_cmps.common_dirs:
            common_paths.append(os.path.join(dir_top, dir_bot))

    mapping = pd.DataFrame(common_paths, columns=["common_dirs"])
    mapping['data_dirs'] = mapping.apply(
        lambda data_row: os.path.join(data_dir, data_row.common_dirs), axis=1)

    mapping['label_dirs'] = mapping.apply(
        lambda data_row: os.path.join(label_dir, data_row.common_dirs), axis=1)

    table = []
    for i, d, l in zip(common_paths, mapping.data_dirs, mapping.label_dirs):
        data_row = []
        data_row += construct_path_lists(d, [".png"])
        data_row += construct_path_lists(l, [".binvox"])
        data_row += [i]
        table.append(data_row)

    paths = pd.DataFrame(table)
    paths.to_csv("paths.csv")
    return paths


def construct_path_lists(data_dir, file_types):
    # print("[construct_path_lists] parsing dir {} for {} ...".format(data_dir, file_types))
    paths = [[] for _ in range(len(file_types))]

    for root, _, files in os.walk(data_dir):
        for f_name in files:
            for i, f_type in enumerate(file_types):
                if f_name.endswith(f_type):
                    (paths[i]).append(root + '/' + f_name)

    if len(file_types) == 1:
        return paths[0]

    return tuple(paths)


if __name__ == '__main__':
    main()
