import os
import sys
import random
import pyglet
import trimesh
import numpy as np
from PIL import Image
import tarfile
import requests


def load_dataset(dataset_name="ShapeNet"):
    if(dataset_name is "ShapeNet"):
        print("[load_dataset] loading {0}".format(dataset_name))

        # second returned value is ignored
        mesh_paths, _ = construct_paths(
            dataset_name, file_types=['.obj', '.mtl'])

        while True:
            try:
                ind = random.randint(0, len(mesh_paths))
                mesh_obj = trimesh.load_mesh(mesh_paths[ind])
            except:
                continue
            break

        if isinstance(mesh_obj, list):
            compund_mesh = mesh_obj.pop(0)
            for m in mesh_obj:
                compund_mesh += m
        else:
            compund_mesh = mesh_obj
        if not os.path.isdir("Renders"):
            write_renders_to_disk(compund_mesh)

        return fetch_renders_from_disk()
    else:
        print("[load_dataset] failed to load {0}".format(dataset_name))


def write_renders_to_disk(mesh, render_dir="./Renders"):
    print("[write_renders_to_disk]")
    scene = mesh.scene()
    N = 5  # random.randint(10, 15)
    if os.path.isdir(render_dir):
        os.system("rm -rf {0}".format(render_dir))

    os.mkdir(render_dir)

    for i in range(N):
        angle = np.radians(random.randint(30, 60))
        axis = random.choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        rotate = trimesh.transformations.rotation_matrix(
            angle, axis, scene.centroid)
        camera_old, _geometry = scene.graph['camera']
        camera_new = np.dot(camera_old, rotate)

        scene.graph['camera'] = camera_new
        pyglet.gl.glDisable(pyglet.gl.GL_CULL_FACE)
        scene.save_image('Renders/render_' + str(i) + '.png')
        pyglet.app.exit()


def fetch_renders_from_disk(render_dir="Renders"):
    if(os.path.isdir(render_dir)):
        print("[fetch_renders_from_disk]")

        ret = []
        for root, _, files in os.walk(render_dir):
            for f_name in files:
                try:
                    im = Image.open(root + '/' + f_name)
                    im = im.resize((512, 512))
                    im = np.array(im)
                    if im.ndim is 3:
                        # remove alpha channel
                        ret.append(im[:, :, 0:3])
                except:
                    pass

        return np.stack(ret)


def construct_paths(data_dir, file_types):
    print("[construct_paths] for dir {}".format(data_dir))
    paths = [[] for _ in range(len(file_types))]

    for root, _, files in os.walk(data_dir):
        for f_name in files:
            for i, f_type in enumerate(file_types):
                if (f_name.endswith(f_type)):
                    (paths[i]).append(root + '/' + f_name)

    return tuple(paths)


def extract_archives(archive_link):
    im_path, im_arc = None, None
    archive_url = requests.get(archive_link, stream=True)
    print(archive_url.headers)
    sys.exit()

    cur_dir = os.listdir()
    if(im_path not in cur_dir):
        if(im_arc not in cur_dir):
            tarfile.open(im_path + '.tar').extractall()
