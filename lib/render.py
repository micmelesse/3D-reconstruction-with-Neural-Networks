import os
import sys
import random
import trimesh
import numpy as np
from lib import utils, dataset
from PIL import Image


def render_dataset(dataset_dir="ShapeNet", num_of_examples=None, render_count=24):
    print("[load_dataset] loading from {0}".format(dataset_dir))

    pathlist_tuple = dataset.construct_path_lists(
        dataset_dir, ['.obj', '.mtl'])
    pathlist = pathlist_tuple[0]  # DANGER, RANDOM
    random.shuffle(pathlist)
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
            render_list.append(fetch_renders_from_disk(renders))
        else:
            write_renders_to_disk(compund_mesh, renders, render_count)
            render_list.append(fetch_renders_from_disk(renders))

    return render_list


def write_renders_to_disk(mesh, renders, render_count=10):
    print("[write_renders_to_disk] writing renders to {0} ... ".format(
        renders))
    # FIXME: stupid but clean
    os.system("rm -rf {}".format(renders))
    utils.make_dir(renders)
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
            '{0}/{1}_{2}.png'.format(renders, os.path.basename(renders), i), resolution=(127, 127))

    return
