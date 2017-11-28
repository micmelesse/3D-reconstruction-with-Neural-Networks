import os
import sys
import random
import pyglet
import trimesh
import numpy as np
from PIL import Image
import tarfile
import requests


def load_dataset(dataset_name=None):
    if(dataset_name is "ShapeNet"):
        print("[load_dataset] loading {0}".format(dataset_name))

        mesh_paths, material_paths = get_ShapeNet_paths()
        while True:
            try:
                ind = random.randint(0, len(mesh_paths))
                mesh_obj = trimesh.load_mesh(mesh_paths[ind])
            except:
                continue
            break

        if (type(mesh_obj) == list):
            compund_mesh = mesh_obj.pop(0)
            for m in mesh_obj:
                compund_mesh += m
        else:
            compund_mesh = mesh_obj

        write_mesh_renders_to_disk(compund_mesh)
        return get_renders()
    else:
        print("[load_dataset] failed to load {0}".format(dataset_name))


def get_renders(render_dir="Renders"):
    if(os.path.isdir(render_dir)):
        print("[get_renders]")

        ret = []
        for root, subdirs, files in os.walk(render_dir):
            for f in files:
                try:
                    im = Image.open(root + '/' + f)
                    im = im.resize((512, 512))
                    im = np.array(im)
                    if im.ndim is 3:
                        # remove alpha channel
                        ret.append(im[:, :, 0:3])
                except:
                    pass

        return np.stack(ret)


def write_mesh_renders_to_disk(mesh, render_dir="./Renders"):
    print("[write_mesh_renders_to_disk]")
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


def get_ShapeNet_paths(mesh_dir='ShapeNet'):
    print("[get_ShapeNet_paths]")
    mesh_paths = []
    material_paths = []

    for root, subdirs, files in os.walk(mesh_dir):
        for f in files:
            if (f.endswith('.obj')):
                mesh_paths.append(root + '/' + f)
            if (f.endswith('.mtl')):
                material_paths.append(root + '/' + f)
    return mesh_paths, material_paths


def extract_archives(archive_link):
    archive_url = requests.get(archive_link, stream=True)
    print(archive_url.headers)
    sys.exit()

    cur_dir = os.listdir()
    if(im_path not in cur_dir):
        if(im_arc not in cur_dir):
            tarfile.open(im_path + '.tar').extractall()
