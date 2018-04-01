import sys
import json
import numpy as np
from keras.utils import to_categorical


x = np.expand_dims(np.load(
    "/Users/micmelesse/Documents/3D-reconstruction-with-neural-networks/out/02691156_10aa040f470500c6a66ef8df4909ded9_x.npy"), 1)
y = to_categorical(np.expand_dims(np.load(
    "/Users/micmelesse/Documents/3D-reconstruction-with-neural-networks/out/02691156_10aa040f470500c6a66ef8df4909ded9_y.npy"), 1)).astype(np.uint8)

x = np.transpose(x, axes=[0, 1, 4, 2, 3])[0:2, :, 0:3, 0:127, 0:127]
# choy et al
params = json.loads(open('params.json').read())
org_dir = params["DIRS"]["CHOY_ET_AL"]
sys.path.insert(0, org_dir)
from lib.solver import Solver
from models.res_gru_net import ResidualGRUNet

net = ResidualGRUNet(compute_grad=False)
net.load("/Users/micmelesse/Documents/3D-reconstruction-with-neural-networks/data/ResidualGRUNet.npy")
solver = Solver(net)
print(x.shape, y.shape)
voxel_prediction, loss, activations = solver.test_output(x, y)
np.save("voxel_prediction.npy", voxel_prediction)
np.save("loss.npy", loss)
np.save("activations.npy", activations)
