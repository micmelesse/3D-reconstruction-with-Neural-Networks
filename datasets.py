import re
import os
import sys
import pyglet
import random
import numpy as np
from PIL import Image
from trimesh import load_mesh


def load_dataset(dataset_dir="./ShapeNet"):
    print("[load_dataset] loading from {0}".format(dataset_dir))
    render_dir = "./ShapeNet_Renders"
    if os.path.isdir(render_dir):
        fetch_renders_from_disk()

    pathlist_tuple = construct_paths(dataset_dir, file_types=['.obj', '.mtl'])

    pathlist = pathlist_tuple[0]
    for mesh_path in pathlist:
        render_path = str.replace(mesh_path, dataset_dir, render_dir)
        try:
            mesh_obj = load_mesh(mesh_path)
            if isinstance(mesh_obj, list):
                compund_mesh = mesh_obj.pop(0)
                for m in mesh_obj:
                    compund_mesh += m
            else:
                compund_mesh = mesh_obj
            print("[load_dataset] succeded in loading {}".format(mesh_path))
            write_renders_to_disk(render_path, compund_mesh, 10)
        except:
            print("[load_dataset] failed to load {}".format(mesh_path))
        

    return


def write_renders_to_disk(render_dir, mesh, N=1):
    print("[write_renders_to_disk] writing to dir {}".format(render_dir))
    scene = mesh.scene()
    if os.path.isdir(render_dir):
        print("[write_renders_to_disk] existing dir  {}?".format(render_dir))
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
        scene.save_image('{}/render_'.format(render_dir) + str(i) + '.png')
        pyglet.app.exit()


def fetch_renders_from_disk(render_dir="./ShapeNet_Renders"):
    if(os.path.isdir(render_dir)):
        print("[fetch_renders_from_disk] in dir {}".format(render_dir))

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
        print("[fetch_renders_from_disk] returning {0} renders from {1}".format(
            len(ret), render_dir))
        return np.stack(ret)


def extract_archives(archive_link):
    im_path, im_arc = None, None
    archive_url = requests.get(archive_link, stream=True)
    print(archive_url.headers)
    sys.exit()

    cur_dir = os.listdir()
    if(im_path not in cur_dir):
        if(im_arc not in cur_dir):
            tarfile.open(im_path + '.tar').extractall()


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
    load_dataset()
