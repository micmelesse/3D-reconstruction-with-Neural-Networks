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

    # init network
    net = network.reconstruction_network()

    # train network
    all_loss = []
    for e in range(net.epoch_count):
        t_start = time.time()  # start timer

        epoch_loss = []
        data_batchs, label_batchs = utils.get_batchs(
            data_all, label_all, net.batch_size)
        try:
            for data, label in zip(data_batchs, label_batchs):
                epoch_loss.append(net.train_step(data, label))
            all_loss.append(epoch_loss)
        except KeyboardInterrupt:
            print("training quit by user after %d seconds" %
                  (time.time()-t_start))
            save_training()
            break

        print("epoch %d took %d seconds" % (e, time.time()-t_start))
        save_training()
