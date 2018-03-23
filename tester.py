import os
import numpy as np
import tensorflow as tf
import lib.dataset as dataset
import lib.network as network
import lib.utils as utils


if __name__ == '__main__':

    EPOCH_DIR = "/Users/micmelesse/Documents/3D-reconstruction-with-neural-networks/out/model_2018-03-20_22:22:52_L:0.1_E:2_B:2/epoch_1"
    PREDICTIONS_DIR = "{}/predictions".format(EPOCH_DIR)
    utils.make_dir(PREDICTIONS_DIR)

    X_test, y_test = dataset.get_model_testset(EPOCH_DIR)

    # network restore to specific epoch 
    net = network.Network()
    net.restore(EPOCH_DIR)

    # split test set into batchs
    X_test_batchs, y_test_batchs = dataset.get_suffeled_batchs(
        X_test, y_test, net.BATCH_SIZE)

    print("testing network ...")
    # test network
    i = 0
    while X_test_batchs and y_test_batchs:
        i += 1
        X = dataset.from_npy(X_test_batchs.popleft())
        y = dataset.from_npy(y_test_batchs.popleft())
        y_hat = net.predict(X)

        utils.vis_multichannel(
            X[0][1], "{}/feature_maps_{}.png".format(PREDICTIONS_DIR, i))
        utils.vis_voxel(
            y[0], "{}/target_{}.png".format(PREDICTIONS_DIR, i))
        utils.vis_voxel(
            y_hat[0], "{}/prediction_{}.png".format(PREDICTIONS_DIR, i))
    print("... done")