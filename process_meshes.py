import os
import random
import trimesh

# default input is sysnet_id for airplanes
def read_ShapeNet(sysnet_id='02691156'):
    meshes = []

    for root, subdirs, files in os.walk('./ShapeNet/'+sysnet_id+'/'):
        for f in files:
            if (f.endswith('.obj')):
                meshes.append(root+'/'+f)

    ri = random.randint(0, len(meshes))
    m = trimesh.load_mesh(meshes[ri])
    m = m[0] if (type(m) is list) else m
    m.show()
