import os
import sys
import random
import trimesh
import numpy as np

# default input is sysnet_id for airplanes


def generate_renders_from_mesh(mesh):
    print("[generate_images_from_mesh]")
    scene = mesh.scene()
    N = random.randint(5, 20)
    for i in range(N):
        angle = np.radians(random.randint(30, 60))
        axis = [random.randint(0, 2), random.randint(
            0, 2), random.randint(0, 2)]
        rotate = trimesh.transformations.rotation_matrix(
            angle, axis, scene.centroid)
        camera_old, _geometry = scene.graph['camera']
        camera_new = np.dot(camera_old, rotate)

        scene.graph['camera'] = camera_new

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
    print("[main]")
    mesh_paths, material_paths = get_ShapeNet_paths()
    print(len(mesh_paths))
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

    meshes.append(compund_mesh)
    generate_renders_from_mesh(meshes[0])


if __name__ == '__main__':
    main()
