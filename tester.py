import os
import numpy as np
import tensorflow as tf
import lib.dataset as dataset
import lib.network as network
import lib.utils as utils


if __name__ == '__main__':
    MODEL_DIR = utils.read_params()['TRAIN_PARAMS']['TEST_DIR']

    net = network.Network()
    net.restore(MODEL_DIR)
    X_test = np.load("{}/X_test.npy".format(os.path.dirname(MODEL_DIR)))
    y_test = np.load("{}/y_test.npy".format(os.path.dirname(MODEL_DIR)))

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

        os.makedirs("{}/tests".format(MODEL_DIR))

        utils.vis_multichannel(
            X[0][1], "{}/epoch_test/feature_maps_{}.png".format(MODEL_DIR, i))
        utils.vis_voxel(
            y[0], "{}/epoch_test/target_{}.png".format(MODEL_DIR, i))
        utils.vis_voxel(
            y_hat[0], "{}/epoch_test/prediction_{}.png".format(MODEL_DIR, i))
    print("... done")
