"""deals with data for project"""
import os
import sys
import random
import tarfile
import pyglet
import trimesh
import binvox_rw
import math
import pandas as pd
import numpy as np
from PIL import Image
from filecmp import dircmp


def main():
    print("test run")
    shapenet = ShapeNet()
    train = shapenet.next_train_batch()
    X = load_dataset(train[:, 0])
    Y = load_labels(train[:, 1])


class ShapeNet:
    def __init__(self):

        self.paths = pd.read_csv("paths.csv", index_col=0).as_matrix()
        np.random.shuffle(self.paths)
        self.N = len(self.paths)
        self.split_index = math.ceil(self.N * 0.8)
        self.train_index = 0
        self.test_index = self.split_index
        self.batch_size = 36

    def next_train_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        prev_index = self.train_index
        self.train_index += batch_size

        if self.train_index >= self.split_index:
            self.train_index = 0
            return None
        else:
            return self.paths[prev_index:self.train_index]

    def next_test_batch(self, batch_size=None):
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


def load_dataset(dataset_path_list):
    if isinstance(dataset_path_list, np.ndarray):
        dataset_path_list = dataset_path_list.tolist()

    return fetch_render_from_disk(dataset_path_list)


def load_labels(label_path_list):
    if isinstance(label_path_list, np.ndarray):
        label_path_list = label_path_list.tolist()

    voxel_list = []
    for voxel_path in label_path_list:
        with open(voxel_path, 'rb') as f:
            voxel_list.append(
                (binvox_rw.read_as_3d_array(f)).data.astype(float))

    return np.stack(voxel_list)


def fetch_render_from_disk(render_path_list):
    png_list = []
    for png_file in render_path_list:
        im = Image.open(png_file)
        im = np.array(im)
        if im.ndim is 3:
            # remove alpha channel
            png_list.append(im[:, :, 0:3])

    return np.stack(png_list)


def render_dataset(dataset_dir="ShapeNet", num_of_examples=None, render_count=24):
    print("[load_dataset] loading from {0}".format(dataset_dir))

    pathlist_tuple = construct_path_lists(
        dataset_dir, file_types=['.obj', '.mtl'])
    pathlist = pathlist_tuple[0]  # DANGER, RANDOM
    random.shuffle(pathlist)
    pathlist = pathlist[:num_of_examples] if num_of_examples is not None else pathlist
    render_list = []

    for mesh_path in pathlist:
        if not os.path.isfile(mesh_path):
            continue
        try:
            mesh_obj = trimesh.load_mesh(mesh_path)
        except Exception as e:
            print("failed to load {}".format(mesh_path))
            continue

        if isinstance(mesh_obj, list):
            compund_mesh = mesh_obj.pop(0)
            for m in mesh_obj:
                compund_mesh += m
        else:
            compund_mesh = mesh_obj

        render_dir = "./ShapeNet_Renders"
        render_path = os.path.dirname(
            str.replace(mesh_path, dataset_dir, render_dir))

        if os.path.isdir(render_path) and os.listdir(render_path) != []:
            render_list.append(fetch_render_from_disk(render_path))
        else:
            write_renders_to_disk(compund_mesh, render_path, render_count)
            render_list.append(fetch_render_from_disk(render_path))

    return render_list


def write_renders_to_disk(mesh, render_path, render_count=10):
    print("[write_renders_to_disk] writing renders to {0} ... ".format(
        render_path))
    # FIXME: stupid but clean
    os.system("rm -rf {}".format(render_path))
    os.makedirs(render_path)
    scene = mesh.scene()
    for i in range(render_count):
        angle = np.radians(random.randint(15, 30))
        axis = random.choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        rotate = trimesh.transformations.rotation_matrix(
            angle, axis, scene.centroid)
        camera_old, _geometry = scene.graph['camera']
        camera_new = np.dot(camera_old, rotate)
        scene.graph['camera'] = camera_new
        # backfaces culled if using original trimesh package
        scene.save_image(
            '{0}/{1}_{2}.png'.format(render_path, os.path.basename(render_path), i), resolution=(127, 127))

    return


def get_common_paths(data_dir, label_dir):

    common_paths = []
    for dir_top, subdir_cmps in dircmp(data_dir, label_dir).subdirs.items():
        for dir_bot in subdir_cmps.common_dirs:
            common_paths.append(os.path.join(dir_top, dir_bot))

    mapping = pd.DataFrame(common_paths, columns=["common_dirs"])
    mapping['data_dirs'] = mapping.apply(
        lambda row: os.path.join(data_dir, row.common_dirs), axis=1)

    mapping['label_dirs'] = mapping.apply(
        lambda row: os.path.join(label_dir, row.common_dirs), axis=1)

    data_paths = []
    label_paths = []
    for d, l in zip(mapping.data_dirs, mapping.label_dirs):
        png_ls = construct_path_lists(d, [".png"])
        binvox_ls = construct_path_lists(l, [".binvox"])
        data_paths += png_ls
        label_paths += binvox_ls * len(png_ls)

    paths = pd.DataFrame(
        {"data_paths": data_paths, "label_paths": label_paths})
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
