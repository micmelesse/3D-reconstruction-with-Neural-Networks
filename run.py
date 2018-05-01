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
import lib.utils as utils
from datetime import datetime


if __name__ == '__main__':
    def save_dataset_split():
        np.save("{}/X_train.npy".format(net.MODEL_DIR), X_train)
        np.save("{}/y_train.npy".format(net.MODEL_DIR), y_train)
        np.save("{}/X_val.npy".format(net.MODEL_DIR), X_val)
        np.save("{}/y_val.npy".format(net.MODEL_DIR), y_val)
        np.save("{}/X_test.npy".format(net.MODEL_DIR), X_test)
        np.save("{}/y_test.npy".format(net.MODEL_DIR), y_test)

    def save_loss(loss_arr, loss_type):
        loss_ndarr = np.array(loss_arr)
        save_dir = net.get_cur_epoch_dir()
        np.save("{}/{}_loss.npy".format(save_dir, loss_type), loss_ndarr)

    def plot_loss(loss_arr, loss_type):
        loss_ndarr = np.array(loss_arr)
        save_dir = net.get_cur_epoch_dir()
        plt.plot(loss_ndarr.flatten())
        plt.savefig("{}/{}_loss.png".format(save_dir, loss_type),
                    bbox_inches="tight")
        plt.close()

    # params on disk
    print(utils.read_params()["TRAIN_PARAMS"])

    # get preprocessed data
    data, label = dataset.load_preprocessed_dataset()

    # init network
    net = network.Network()
    # net.init_parameters()

    params = net.get_params()
    train_params = params["TRAIN_PARAMS"]

    # split dataset
    X_train, y_train, X_val, y_val, X_test, y_test = dataset.train_val_test_split(
        data, label)
    save_dataset_split()

    print("training loop")
    # train network
    train_loss, val_loss, test_loss = [], [], []
    for e in range(train_params["EPOCH_COUNT"]):
        t_start = time.time()  # timer for epoch
        net.create_epoch_dir()

        # training and validaiton loops
        try:
            # split traning and validation set into batchs
            X_train_batchs, y_train_batchs = dataset.shuffle_batchs(
                X_train, y_train, train_params["BATCH_SIZE"])
            X_val_batchs, y_val_batchs = dataset.shuffle_batchs(
                X_val, y_val, train_params["BATCH_SIZE"])

            if params["TRAIN_PARAMS"]["VALIDATION_INTERVAL"]:
                val_interval = params["TRAIN_PARAMS"]["VALIDATION_INTERVAL"]
            else:
                val_interval = math.ceil(len(X_train_batchs)/len(X_val_batchs))

                # train step
            counter = 0
            epoch_train_loss, epoch_val_loss = [], []
            while X_train_batchs and y_train_batchs:
                counter += 1
                if X_val_batchs and counter >= val_interval:
                    counter = 0
                    X = X_val_batchs.popleft()
                    y = y_val_batchs.popleft()
                    epoch_val_loss.append(net.step(X, y, 'val'))
                else:
                    X = X_train_batchs.popleft()
                    y = y_train_batchs.popleft()
                    if params["MODE"] == "DEBUG":
                        epoch_train_loss.append(net.step(X, y, 'debug'))
                    else:
                        epoch_train_loss.append(net.step(X, y, 'train'))
            train_loss.append(epoch_train_loss)
            val_loss.append(epoch_val_loss)

        except KeyboardInterrupt:
            print("training quit by user after %d seconds" %
                  (time.time()-t_start))
            train_loss.append(epoch_train_loss)
            val_loss.append(epoch_val_loss)
            net.save()
            save_loss(train_loss, 'train')
            save_loss(val_loss, 'val')
            exit()

        print("epoch %d took %d seconds to train" % (e, time.time()-t_start))
        net.save()
        save_loss(train_loss, 'train')
        save_loss(val_loss, 'val')
        plot_loss(train_loss, 'train')
        plot_loss(val_loss, 'val')
