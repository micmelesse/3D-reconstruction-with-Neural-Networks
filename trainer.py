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

data_all = np.array(sorted(path.construct_path_lists("out", "data_")))
label_all = np.array(sorted(path.construct_path_lists("out", "labels_")))
net = network.R2N2()

model_dir = "out/model_{}_L:{}_E:{}_B:{}".format(
    net.create_time, net.learn_rate, net.epoch_count, net.batch_size)
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

# train network
print("training start")
for e in range(net.epoch_count):
    epoch_loss = []
    data_batchs, label_batchs = utils.get_batchs(
        data_all, label_all, net.batch_size)
    t_start = time.time()
    try:
        for b, (data, label) in enumerate(zip(data_batchs, label_batchs)):
            epoch_loss.append(net.train_step(data, label))
    except KeyboardInterrupt:
        print("training quit by user")
        net.session_loss.append(epoch_loss)
        net.save("{}/epoch_{:04d}".format(model_dir, e))
        print("training quit after %d seconds" % (time.time()-t_start))
        break

    net.session_loss.append(epoch_loss)
    net.save("{}/epoch_{:04d}".format(model_dir, e))
    print("epoch %d took %d seconds" % (e, time.time()-t_start))
