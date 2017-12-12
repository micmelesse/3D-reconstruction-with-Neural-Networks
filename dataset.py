"""deals with data for project"""
import os
import sys
import random
import tarfile
import pyglet
import trimesh
import numpy as np
from PIL import Image


def load_dataset(dataset_dir="./ShapeNet", N=None):
    print("[load_dataset] loading from {0}".format(dataset_dir))

    ret = []
    pathlist_tuple = construct_paths(dataset_dir, file_types=['.obj', '.mtl'])
    pathlist = pathlist_tuple[0]
    pathlist = pathlist[:N] if N is not None else pathlist
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
        render_path_tuple = os.path.split(
            str.replace(mesh_path, dataset_dir, render_dir))

        if os.path.isdir(render_path_tuple[0]) and os.listdir(render_path_tuple[0]) != []:
            ret.append(fetch_renders_from_disk(render_path_tuple[0]))
        else:
            write_renders_to_disk(compund_mesh, render_path_tuple, 10)
            ret.append(fetch_renders_from_disk(render_path_tuple[0]))

    return ret


def write_renders_to_disk(mesh, render_path_tuple, N=5):
    # FIXME: stupid but clean
    os.system("rm -rf {}".format(render_path_tuple[0]))
    os.makedirs(render_path_tuple[0])
    scene = mesh.scene()
    for i in range(N):
        angle = np.radians(random.randint(30, 60))
        axis = random.choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        rotate = trimesh.transformations.rotation_matrix(
            angle, axis, scene.centroid)
        camera_old, _geometry = scene.graph['camera']
        camera_new = np.dot(camera_old, rotate)
        scene.graph['camera'] = camera_new
        # backfaces culled if using original trimesh package
        scene.save_image(
            '{0}/{1}_{2}.png'.format(render_path_tuple[0], os.path.splitext(render_path_tuple[1]), i))


def fetch_renders_from_disk(render_path):
    ret = []
    for root, _, files in os.walk(render_path):
        for f_name in files:
            try:
                im = Image.open(root + '/' + f_name)
                im = im.resize((512, 512))
                im = np.array(im)
                if im.ndim is 3:
                    # remove alpha channel
                    ret.append(im[:, :, 0:3])
            except Exception as e:
                print(e)
                continue
    return np.stack(ret)


def construct_paths(data_dir, file_types):
    print("[construct_paths] parsing dir {} for {}".format(data_dir, file_types))
    paths = [[] for _ in range(len(file_types))]

    for root, _, files in os.walk(data_dir):
        for f_name in files:
            for i, f_type in enumerate(file_types):
                if f_name.endswith(f_type):
                    (paths[i]).append(root + '/' + f_name)
    return tuple(paths)


if __name__ == '__main__':
    load_dataset(N=5)
