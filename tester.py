import os
import numpy as np
import tensorflow as tf
import lib.dataset as dataset
import lib.network as network
import lib.utils as utils


if __name__ == '__main__':
    model_dir = "/Users/micmelesse/Documents/3D-reconstruction-with-neural-networks/out/model_2018-03-14_23:14:16_L:0.1_E:2_B:2/epoch_1"

    net = network.Network()
    net.restore(model_dir)
    X_test = np.load("{}/X_test.npy".format(os.path.dirname(model_dir)))
    y_test = np.load("{}/y_test.npy".format(os.path.dirname(model_dir)))

    # split test set into batchs
    X_test_batchs, y_test_batchs = dataset.get_suffeled_batchs(
        X_test, y_test, net.batch_size)

    print("testing network")
    # test network
    i = 0
    while X_test_batchs and y_test_batchs:
        i += 1
        X = dataset.from_npy(X_test_batchs.popleft())
        y = dataset.from_npy(y_test_batchs.popleft())
        y_hat = net.predict(X)

        os.makedirs("{}/epoch_test".format(model_dir))

        utils.vis_multichannel(
            X[0][1], "{}/epoch_test/feature_maps_{}.png".format(model_dir, i))
        utils.vis_voxel(
            y[0], "{}/epoch_test/target_{}.png".format(model_dir, i))
        utils.vis_voxel(
            y_hat[0], "{}/epoch_test/prediction_{}.png".format(model_dir, i))
