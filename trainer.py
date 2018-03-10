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

    def save_loss(loss_arr, loss_type):
        save_dir = net.get_save_dir()
        np.save("{}/{}_loss.npy".format(save_dir, loss_type), loss_arr)
        tf.train.Saver().save(net.sess, "{}/model.ckpt".format(save_dir))
        plt.plot(np.array(loss_arr).flatten())
        plt.savefig("{}/plot_{}_loss.png".format(save_dir, loss_type),
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
    train_loss = []
    val_loss = []
    for e in range(net.epoch_count):
        t_start = time.time()  # timer for epoch

        # split trainig set in to  validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2)

        # training and validaiton loops
        try:
             # split traning set into batchs
            X_train_batchs, y_train_batchs = dataset.get_batchs(
                X_train, y_train, net.batch_size)

            # train step
            epoch_train_loss = []
            for X, y in zip(X_train_batchs, y_train_batchs):
                epoch_train_loss.append(net.train_step(X, y))
            train_loss.append(epoch_train_loss)

            # split validation set into batchs
            X_val_batchs, y_val_batchs = dataset.get_batchs(
                X_val, y_val, net.batch_size)

            # validation step
            epoch_val_loss = []
            for X, y in zip(X_val_batchs, y_val_batchs):
                epoch_val_loss.append(net.val_step(X, y))
            val_loss.append(epoch_val_loss)

        except KeyboardInterrupt:
            print("training quit by user after %d seconds" %
                  (time.time()-t_start))
            save_loss(train_loss, 'train')
            save_loss(val_loss, 'val')
            break

        print("epoch %d took %d seconds" % (e, time.time()-t_start))
        save_loss(train_loss, 'train')
        save_loss(val_loss, 'val')

    # split test set into batchs
    X_test_batchs, y_test_batchs = dataset.get_batchs(
        X_test, y_test, net.batch_size)

    print("testing network")
    # test network
    test_loss = []
    for X, y in zip(X_test_batchs, y_test_batchs):
        test_loss.append(net.val_step(X, y))
    save_loss(val_loss, 'test')
