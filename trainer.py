import os
import sys
import math
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import lib.dataset as dataset
import lib.network as network
import lib.path as path
import lib.utils as utils
from datetime import datetime

from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    def save_training():
        save_dir = net.get_save_dir()
        np.save("{}/trainloss.npy".format(save_dir), all_loss)
        tf.train.Saver().save(net.sess, "{}/model.ckpt".format(save_dir))
        plt.plot(np.array(all_loss).flatten())
        plt.savefig("{}/plot_trainloss.png".format(save_dir),
                    bbox_inches='tight')
        plt.close()

    # get data
    data_all = np.array(sorted(path.construct_path_lists("out", "data_")))
    label_all = np.array(sorted(path.construct_path_lists("out", "labels_")))

    # split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        data_all, label_all, test_size=0.2)

    # init network
    net = network.Network()

    # train network
    all_loss = []
    for e in range(net.epoch_count):
        epoch_loss = []  # the loss per batch for this epoch
        t_start = time.time()  # start timer

        # split trainig set in to  validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2)

        # split traning set into batchs
        X_batchs, y_batchs = dataset.get_batchs(
            X_train, y_train, net.batch_size)
        try:
            for X, y in zip(X_batchs, y_batchs):
                epoch_loss.append(net.train_step(X, y))
            all_loss.append(epoch_loss)
        except KeyboardInterrupt:
            print("training quit by user after %d seconds" %
                  (time.time()-t_start))
            save_training()
            break

        print("epoch %d took %d seconds" % (e, time.time()-t_start))
        save_training()
