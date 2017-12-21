"""deals with data for project"""
import os
import sys
import random
import tarfile
import pyglet
import trimesh
import binvox_rw
import numpy as np
from PIL import Image


def load_dataset(dataset_dir="ShapeNetRendering", num_of_examples=None):
    return fetch_renders_from_disk(dataset_dir, num_of_examples)


def load_labels(dataset_dir="ShapeNetVox32", num_of_examples=None):
    pathlist_tuple = construct_path_lists(
        dataset_dir, file_types=['.binvox'])
    pathlist = pathlist_tuple[0]
    random.shuffle(pathlist)
    pathlist = pathlist[:num_of_examples] if num_of_examples is not None else pathlist
    voxel_list = []

    for voxel_path in pathlist:
        with open(voxel_path, 'rb') as f:
            voxel_list.append(
                (binvox_rw.read_as_3d_array(f)).data.astype(float))

    return np.stack(voxel_list)


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
            render_list.append(fetch_renders_from_disk(render_path))
        else:
            write_renders_to_disk(compund_mesh, render_path, render_count)
            render_list.append(fetch_renders_from_disk(render_path))

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


def fetch_renders_from_disk(render_path="ShapeNetRendering", num_of_examples=None):
    print("[fetch_renders_from_disk] fetching renders from {0} ... ".format(
        render_path))

    pathlist_tuple = construct_path_lists(render_path, file_types=['.png'])
    pathlist = pathlist_tuple[0]
    random.shuffle(pathlist)
    pathlist = pathlist[:num_of_examples] if num_of_examples is not None else pathlist
    png_list = []

    for png_file in pathlist:
        try:
            im = Image.open(png_file)
            im = np.array(im)
            if im.ndim is 3:
                # remove alpha channel
                png_list.append(im[:, :, 0:3])
        except Exception as e:
            print("[fetch_renders_from_disk] failed on {0} ... ".format(
                render_path))
            continue

    return np.stack(png_list)


def construct_path_lists(data_dir, file_types):
    #print("[construct_path_lists] parsing dir {} for {} ...".format(data_dir, file_types))
    paths = [[] for _ in range(len(file_types))]

    for root, _, files in os.walk(data_dir):
        for f_name in files:
            for i, f_type in enumerate(file_types):
                if f_name.endswith(f_type):
                    (paths[i]).append(root + '/' + f_name)

    if len(file_types) == 1:
        return paths[0]

    return tuple(paths)
