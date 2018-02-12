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
import lib.params as params
import lib.network as network
import lib.path as path
import lib.utils as utils
from datetime import datetime

data_all = np.array(sorted(path.construct_path_lists("out", "data_")))
label_all = np.array(sorted(path.construct_path_lists("out", "labels_")))
net = network.R2N2()

model_dir = "out/model_{}_{}_{} {}".format(
    net.learn_rate, net.epoch, net.batch_size, net.create_time)

# train network
print("training start")
loss_all = []
for e in range(net.epoch):
    start_time = time.time()
    loss_epoch = []
    batch_number = 0
    data_batchs, label_batchs = utils.get_batchs(
        data_all, label_all, net.batch_size)
    for data, label in zip(data_batchs, label_batchs):
        loss_epoch.append(net.train_step(data, label))
        print("epoch_{:03d}-batch_{:03d}: loss={}".format(e,
                                                          batch_number, loss_epoch[-1]))
        batch_number += 1
    loss_all.append(loss_epoch)
    net.save("{}/epoch_{:03d}".format(model_dir, e), "loss", loss_all)
    print("epoch %d took %d seconds" % (e, time.time()-start_time))
