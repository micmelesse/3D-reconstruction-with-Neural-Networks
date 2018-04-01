import sys
import json
import numpy as np
params = json.loads(open('params.json').read())
org_dir = params["DIRS"]["CHOY_ET_AL"]
sys.path.insert(0, org_dir)
from lib.solver import Solver
from models.res_gru_net import ResidualGRUNet

x = np.transpose(np.expand_dims(np.load(
    "/Users/micmelesse/Documents/3D-reconstruction-with-neural-networks/out/02691156_10155655850468db78d106ce0a280f87_x.npy"), 1), [0, 1, 4, 2, 3])[:, :, 0:3, 0:127, 0:127]
y = np.transpose(np.expand_dims(np.load(
    "/Users/micmelesse/Documents/3D-reconstruction-with-neural-networks/out/02691156_10155655850468db78d106ce0a280f87_y.npy"), 0), [0, 1, 4, 2, 3])

print(x.shape, y.shape)
net = ResidualGRUNet(compute_grad=False)
net.load("/Users/micmelesse/Documents/3D-reconstruction-with-neural-networks/data/ResidualGRUNet.npy")
solver = Solver(net)
voxel_prediction, _ = solver.test_output(x)
np.save("voxel_prediction.npy", voxel_prediction)
# np.save("loss.npy", loss)
# np.save("activations.npy", activations)
