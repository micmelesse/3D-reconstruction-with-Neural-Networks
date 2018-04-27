"""deals with data for project"""
import re
import json
import os
import sys
import math
import trimesh
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
from numpy.random import randint, permutation, shuffle
from numpy import radians
from natsort import natsorted


def load_obj_id(obj_id):
    data_path, label_path = id_to_path(obj_id)
    return load_imgs_from_dir(data_path), np.squeeze(load_voxs_from_dir(label_path))


def id_to_path(obj_id, data_dir="./data/ShapeNetRendering/", label_dir="./data/ShapeNetVox32/"):
    regex = re.search("(.*)_(.*)", obj_id)
    ret_1 = os.path.join(data_dir, regex.group(1), regex.group(2))
    ret_2 = os.path.join(label_dir, regex.group(1), regex.group(2))
    return ret_1, ret_2


# loading functions
def load_img(img_path):
    return np.array(Image.open(img_path))


def load_vox(vox_path):
    with open(vox_path, 'rb') as f:
        return to_categorical(binvox_rw.read_as_3d_array(f).data)


def load_imgs(img_path_list):
    assert(isinstance(img_path_list, (list, np.ndarray)))

    ret = []
    for p in img_path_list:
        ret.append(load_img(p))
    return np.stack(ret)


def load_voxs(vox_path_list):
    assert(isinstance(vox_path_list, (list, np.ndarray)))

    ret = []
    for p in vox_path_list:
        ret.append(load_vox(p))
    return np.stack(ret)


def load_imgs_from_dir(img_dir):
    img_path_list = construct_file_path_list_from_dir(img_dir, [".png"])
    return load_imgs(img_path_list)


def load_voxs_from_dir(vox_dir):
    vox_path_list = construct_file_path_list_from_dir(vox_dir, [".binvox"])
    return load_voxs(vox_path_list)


# #  dataset loading functions
def load_data(data_samples):
    if isinstance(data_samples, str):
        data_samples = [data_samples]
    return load_imgs(data_samples)


def load_label(label_samples):
    if isinstance(label_samples, str):
        label_samples = [label_samples]
    return np.squeeze(load_voxs(label_samples))


# load preprocessed data and labels
def load_preprocessed_dataset():
    data_preprocessed_dir = utils.read_params(
    )["DIRS"]["DATA_PREPROCESSED"]

    data_all = sorted(
        dataset.construct_file_path_list_from_dir(data_preprocessed_dir, ["_x.npy"]))
    label_all = sorted(
        dataset.construct_file_path_list_from_dir(data_preprocessed_dir, ["_y.npy"]))

    return np.array(data_all), np.array(label_all)


def load_random_sample():
    data, label = load_preprocessed_dataset()
    i = randint(0, len(data))
    return np.load(data[i]), np.load(label[i])


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


def shuffle_batchs(data, label, batch_size):
    # print(data, label, batch_size)
    assert(len(data) == len(label))
    num_of_batches = math.ceil(len(data)/batch_size)
    perm = permutation(len(data))
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


def setup_dir():
    params = utils.read_params()
    DIR = params["DIRS"]
    for d in DIR.values():
        utils.make_dir(d)

    param_name = "params.json"
    if not os.path.exists(param_name):
        param_data = {
            "MODE": "TRAIN",
            "TRAIN_PARAMS": {
            },
            "AWS_PARAMS": {
            },
            "DIRS": {
            }
        }
        with open(param_name, 'w') as param_file:
            json.dump(param_data, param_file)


def construct_file_path_list_from_dir(dir, file_filter):
    if isinstance(file_filter, str):
        file_filter = [file_filter]
    paths = [[] for _ in range(len(file_filter))]

    for root, _, files in os.walk(dir):
        for f_name in files:
            for i, f_substr in enumerate(file_filter):
                if f_substr in f_name:
                    (paths[i]).append(root + '/' + f_name)

    for i, p in enumerate(paths):
        paths[i] = natsorted(p)

    if len(file_filter) == 1:
        return paths[0]

    return tuple(paths)


def create_path_csv(data_dir, label_dir):
    print("creating path csv for {} and {}".format(data_dir, label_dir))
    params = utils.read_params()

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
        data_row += construct_file_path_list_from_dir(d, [".png"])
        data_row += construct_file_path_list_from_dir(l, [".binvox"])
        table.append(data_row)

    paths = pd.DataFrame(table)
    paths.to_csv("{}/paths.csv".format(params["DIRS"]["OUTPUT"]))
    return paths


def download_from_link(link):
    download_folder = os.path.splitext(os.path.basename(link))[0]
    archive = download_folder + ".tgz"

    if not os.path.isfile(archive):
        os.system('wget -c {0}'.format(link))

    os.system("tar -xvzf {0}".format(archive))
    os.rename(download_folder, "data/{}".format(download_folder))
    os.system("rm -f {0}".format(archive))


def download_dataset():
    LABEL_LINK = 'ftp://cs.stanford.edu/cs/cvgl/ShapeNetVox32.tgz'
    DATA_LINK = "ftp://cs.stanford.edu/cs/cvgl/ShapeNetRendering.tgz"

    if not os.path.isdir("data/ShapeNetVox32"):
        download_from_link(LABEL_LINK)

    if not os.path.isdir("data/ShapeNetRendering"):
        download_from_link(DATA_LINK)


def preprocess_dataset():
    params = utils.read_params()
    dataset_size = params["MISC"]["DATASET_SIZE"]
    output_dir = params["DIRS"]["OUTPUT"]
    data_preprocessed_dir = params["DIRS"]["DATA_PREPROCESSED"]
    data_dir = params["DIRS"]["DATA"]

    if not os.path.isfile("{}/paths.csv".format(output_dir)):
        dataset.create_path_csv(
            "{}/ShapeNetRendering".format(data_dir), "{}/ShapeNetVox32".format(data_dir))

    path_list = pd.read_csv(
        "{}/paths.csv".format(output_dir), index_col=0).as_matrix()
    # randomly pick examples from dataset
    shuffle(path_list)

    if dataset_size <= 0 or dataset_size >= len(path_list):
        dataset_size = len(path_list)

    for i in range(dataset_size):
        model_name = path_list[i, 0]
        utils.to_npy('{}/{}_x'.format(data_preprocessed_dir, model_name),
                     load_data(path_list[i, 1:-1]))
        utils.to_npy('{}/{}_y'.format(data_preprocessed_dir, model_name),
                     load_label(path_list[i, -1]))


def render_dataset(dataset_dir="ShapeNet", num_of_examples=None, render_count=24):
    print("[load_dataset] loading from {0}".format(dataset_dir))

    pathlist_tuple = construct_file_path_list_from_dir(
        dataset_dir, ['.obj', '.mtl'])
    pathlist = pathlist_tuple[0]  # DANGER, RANDOM
    pathlist = pathlist[:num_of_examples] if num_of_examples is not None else pathlist
    render_list = []

    for mesh_path in pathlist:
        if not os.path.isfile(mesh_path):
            continue
        try:
            mesh_obj = trimesh.load_mesh(mesh_path)
        except:
            print("failed to load {}".format(mesh_path))
            continue

        if isinstance(mesh_obj, list):
            compund_mesh = mesh_obj.pop(0)
            for m in mesh_obj:
                compund_mesh += m
        else:
            compund_mesh = mesh_obj

        render_dir = "./ShapeNet_Renders"
        renders = os.path.dirname(
            str.replace(mesh_path, dataset_dir, render_dir))

        if os.path.isdir(renders) and os.listdir(renders) != []:
            render_list.append(load_imgs_from_dir(renders))
        else:
            write_renders_to_disk(compund_mesh, renders, render_count)
            render_list.append(load_imgs_from_dir(renders))

    return render_list


def write_renders_to_disk(mesh, renders, render_count=10):
    print("[write_renders_to_disk] writing renders to {0} ... ".format(
        renders))
    # FIXME: stupid but clean
    os.system("rm -rf {}".format(renders))
    utils.make_dir(renders)
    scene = mesh.scene()
    for i in range(render_count):
        angle = radians(random.randint(15, 30))
        axis = random.choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        rotate = trimesh.transformations.rotation_matrix(
            angle, axis, scene.centroid)
        camera_old, _geometry = scene.graph['camera']
        camera_new = np.dot(camera_old, rotate)
        scene.graph['camera'] = camera_new
        # backfaces culled if using original trimesh package
        scene.save_image(
            '{0}/{1}_{2}.png'.format(renders, os.path.basename(renders), i), resolution=(127, 127))

    return
