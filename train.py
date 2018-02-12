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

# read params
with open("config/train.params") as f:
    learning_rate = float(params.read_param(f.readline()))
    batch_size = int(params.read_param(f.readline()))
    epoch = int(params.read_param(f.readline()))

    print("training with a learning rate of {} for {} epochs with batchs of size {}".format(
        learning_rate, epoch, batch_size))


data_all = np.array(sorted(path.construct_path_lists("out", "data_")))
label_all = np.array(sorted(path.construct_path_lists("out", "labels_")))
# print(data_all.shape,label_all.shape)
net = network.R2N2(learning_rate)

model_dir = "out/model_{}_{}_{} {}".format(
    learning_rate, epoch, batch_size, datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))

# train network
loss_all = []
acc_all = []
print("training start")
for e in range(epoch):
    start_time = time.time()
    loss_epoch = []
    batch_number = 0
    data_batchs, label_batchs = utils.get_batchs(
        data_all, label_all, batch_size)
    for data, label in zip(data_batchs, label_batchs):
        loss_epoch.append(net.train_step(data, label))
        print("epoch_{:03d}-batch_{:03d}: loss={}".format(e,
                                                          batch_number, loss_epoch[-1]))
        batch_number += 1
    loss_all.append(loss_epoch)
    net.save("{}/epoch_{:03d}".format(model_dir, e), "loss", loss_all)
    print("epoch %d took %d seconds" % (e, time.time()-start_time))
