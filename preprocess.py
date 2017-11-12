import os
import sys
import random
import trimesh
import requests
import tarfile
import zipfile
import numpy as np
import pyglet.gl as gl
from PIL import Image
from urllib.request import urlretrieve


def constuct_input_matrix(im_dir):
    ret = []
    for root, subdirs, files in os.walk(im_dir):
        for f in files:
            try:
                im = Image.open(root + '/' + f)
                im = im.resize((512, 512))
                im = np.array(im)
                if im.ndim is 3:
                    # remove alpha channel
                    ret.append(im[:, :, 0:3]);
            except:
                pass

    return np.stack(ret)


def extract_archives(archive_link):
    archive_url = requests.get(archive_link, stream=True)
    print(archive_url.headers)
    sys.exit()

    cur_dir = os.listdir()
    if(im_path not in cur_dir):
        if(im_arc not in cur_dir):
            tarfile.open(im_path + '.tar').extractall()


def generate_renders_from_mesh(mesh):
    print("[generate_images_from_mesh]")
    os.system("rm -rf Renders/*")
    scene = mesh.scene()
    N = random.randint(5, 15)
    for i in range(N):
        angle = np.radians(random.randint(30, 60))
        axis = random.choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        rotate = trimesh.transformations.rotation_matrix(
            angle, axis, scene.centroid)
        camera_old, _geometry = scene.graph['camera']
        camera_new = np.dot(camera_old, rotate)

        scene.graph['camera'] = camera_new
        gl.glDisable(gl.GL_CULL_FACE)
        scene.save_image('Renders/render_' + str(i) + '.png')



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


def main():
    print("[trimesh_main]")
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

    generate_renders_from_mesh(compund_mesh)


if __name__ == '__main__':
    main()
