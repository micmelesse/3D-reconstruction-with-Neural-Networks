"""deals with data for project"""
import re
import os
import sys
import math
import random
import tarfile
import numpy as np
import pandas as pd
from PIL import Image
from filecmp import dircmp
from collections import deque
from third_party import binvox_rw
from lib import utils, dataset
from sklearn import model_selection
from keras.utils import to_categorical


def load_obj_id(obj_id):
    data_path, label_path = id_to_path(obj_id)
    return load_imgs_from_dir(data_path), np.squeeze(load_voxs_from_dir(label_path))


def id_to_path(obj_id, data_dir="./data/ShapeNetRendering/", label_dir="./data/ShapeNetVox32/"):
    regex = re.search("(.*)_(.*)", obj_id)
    ret_1 = os.path.join(data_dir, regex.group(1), regex.group(2))
    ret_2 = os.path.join(label_dir, regex.group(1), regex.group(2))
    return ret_1, ret_2


# img loading functions
def load_img(img_path):
    return np.array(Image.open(img_path))


def load_imgs(img_path_list):
    assert(isinstance(img_path_list, list))

    ret = []
    for p in img_path_list:
        ret.append(load_img(p))
    return np.stack(ret)


def load_imgs_from_dir(img_dir):
    img_path_list = construct_path_lists(img_dir, [".png"])
    return load_imgs(img_path_list)


# voxel loading functions
def load_vox(vox_path):
    with open(vox_path, 'rb') as f:
        return to_categorical(binvox_rw.read_as_3d_array(f).data)


def load_voxs(vox_path_list):
    assert(isinstance(vox_path_list, list))

    ret = []
    for p in vox_path_list:
        ret.append(load_vox(p))
    return np.stack(ret)


def load_voxs_from_dir(vox_dir):
    vox_path_list = construct_path_lists(vox_dir, [".binvox"])
    return load_voxs(vox_path_list)


#  dataset loading functions
def load_data(data_samples):
    if isinstance(data_samples, str):
        data_samples = [data_samples]

    ret = []
    for data_path in data_samples:
        data_row = load_imgs(data_path)
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


# get data and labels
def load_preprocessed_dataset():
    data_all = sorted(dataset.construct_path_lists("out", ["_x.npy"]))
    label_all = sorted(dataset.construct_path_lists("out", ["_y.npy"]))
    return np.array(data_all), np.array(label_all)


def load_preprocessed_sample():
    data_all = sorted(dataset.construct_path_lists("out", ["_x.npy"]))
    label_all = sorted(dataset.construct_path_lists("out", ["_y.npy"]))
    i = np.random.randint(0, len(data_all))
    return np.load(data_all[i]), np.load(label_all[i])


def load_testset(model_dir):
    try:
        X_test = np.load(
            "{}/X_test.npy".format(model_dir))
        y_test = np.load(
            "{}/y_test.npy".format(model_dir))
    except:
        model_dir = os.path.dirname(model_dir)
        X_test = np.load(
            "{}/X_test.npy".format(model_dir))
        y_test = np.load(
            "{}/y_test.npy".format(model_dir))

    return X_test, y_test


def to_npy(out_dir, arr):
    np.save(out_dir, arr)


def from_npy(npy_path):
    if isinstance(npy_path, str):
        return np.expand_dims(np.load(npy_path), 0)
    ret = []
    for p in npy_path:
        ret.append(np.load(p))
    return np.stack(ret)


def convert_dataset_to_npy(paths, N=None):
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


def construct_path_lists(data_dir, file_filter):
    if isinstance(file_filter, str):
        file_filter = [file_filter]
    paths = [[] for _ in range(len(file_filter))]

    for root, _, files in os.walk(data_dir):
        for f_name in files:
            for i, f_substr in enumerate(file_filter):
                if f_substr in f_name:
                    (paths[i]).append(root + '/' + f_name)

    for p in paths:
        p.sort()
    if len(file_filter) == 1:
        return paths[0]

    return tuple(paths)


def write_path_csv(data_dir, label_dir):
    print("creating path csv for {} and {}".format(data_dir, label_dir))

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
    for n, d, l in zip(common_paths, mapping.data_dirs, mapping.label_dirs):
        data_row = [os.path.dirname(n)+"_"+os.path.basename(n)]
        data_row += construct_path_lists(d, [".png"])
        data_row += construct_path_lists(l, [".binvox"])
        table.append(data_row)

    paths = pd.DataFrame(table)
    paths.to_csv("out/paths.csv")
    return paths


def download_dataset(link):
    download_folder = os.path.splitext(os.path.basename(link))[0]
    archive = download_folder + ".tgz"

    if not os.path.isfile(archive):
        os.system('wget -c {0}'.format(link))

    os.system("tar -xvzf {0}".format(archive))
    os.rename(download_folder, "data/{}".format(download_folder))
    os.system("rm -f {0}".format(archive))


def download():
    LABEL_LINK = 'ftp://cs.stanford.edu/cs/cvgl/ShapeNetVox32.tgz'
    DATA_LINK = "ftp://cs.stanford.edu/cs/cvgl/ShapeNetRendering.tgz"

    if not os.path.isdir("data/ShapeNetVox32"):
        download_dataset(LABEL_LINK)

    if not os.path.isdir("data/ShapeNetRendering"):
        download_dataset(DATA_LINK)


def main():
    example_count = utils.read_params()['TRAIN_PARAMS']['SAMPLE_SIZE']
    if not os.path.isfile("out/paths.csv"):
        dataset.write_path_csv("data/ShapeNetRendering", "data/ShapeNetVox32")
    convert_dataset_to_npy(read_paths(), N=example_count)


if __name__ == '__main__':
    main()
