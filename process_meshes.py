import os
import sys
import random
import trimesh
import numpy as np

# default input is sysnet_id for airplanes
def generate_images_from_mesh(mesh):
    scene = mesh.scene()
    rotate = trimesh.transformations.rotation_matrix(np.radians(45.0), [0, 1, 0],
                                                     scene.centroid)
    for i in range(4):
        trimesh.constants.log.info('Saving image %d', i)
        camera_old, _geometry = scene.graph['camera']
        camera_new = np.dot(camera_old, rotate)
        scene.graph['camera'] = camera_new
        file_name = 'render_' + str(i) + '.png'

        if not '-nw' in sys.argv:
            scene.save_image(file_name,
                             resolution=np.array([1920, 1080]) * 2)

def read_ShapeNet(mesh_path='ShapeNet'):
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
    # trimesh.scene.SceneViewer.toggle_culling()
    meshes, materials = read_ShapeNet()
    mesh = meshes[10]
    mesh_list = trimesh.load_mesh(mesh)
    compund_mesh = mesh_list.pop(0)
    for m in mesh_list:
        compund_mesh += m

    generate_images_from_mesh(compund_mesh)
