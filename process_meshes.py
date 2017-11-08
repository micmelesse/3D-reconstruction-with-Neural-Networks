import os
import sys
import random
import trimesh
import numpy as np

# default input is sysnet_id for airplanes


def generate_images_from_mesh(mesh):
    print("[generate_images_from_mesh]")
    scene = mesh.scene();
    N = random.randint(5, 20);
    for i in range(N):
        angle = np.radians(random.randint(30, 60))
        axis = [random.randint(0, 2), random.randint(
            0, 2), random.randint(0, 2)]
        rotate = trimesh.transformations.rotation_matrix(
            angle, axis, scene.centroid)
        camera_old, _geometry = scene.graph['camera'];
        camera_new = np.dot(camera_old, rotate);

        print (scene.graph)
        sys.exit()
        scene.graph['camera'] = camera_new;

        scene.save_image('Renders/render_' + str(i) + '.png');


def read_ShapeNet(mesh_path='ShapeNet'):
    print("[read_ShapeNet]")
    meshes = []
    materials = []

    for root, subdirs, files in os.walk(mesh_path):
        for f in files:
            if (f.endswith('.obj')):
                meshes.append(root + '/' + f)

            if (f.endswith('.mtl')):
                materials.append(root + '/' + f)

    return meshes, materials


if __name__ == '__main__':
    meshes, materials = read_ShapeNet()
    mesh = meshes[random.randint(1, len(meshes))]
    mesh_obj = trimesh.load_mesh(mesh)

    if (type(mesh_obj)==list):
        compund_mesh = mesh_obj.pop(0)
        for m in mesh_obj:
            compund_mesh += m
    else:
        compund_mesh=mesh_obj

    generate_images_from_mesh(compund_mesh)
