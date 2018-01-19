
# coding: utf-8

# In[10]:


import re
import os
import sys
import math
import time
import utils
import random
import layers
import dataset
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from network import network


# In[7]:


#read params
regex="^.*=(.*)$"
with open("train.params") as f:
    learning_rate=float(re.findall(regex,f.readline())[0])
    batch_size=int(re.findall(regex,f.readline())[0])
    epoch=int(re.findall(regex,f.readline())[0])
   
    print("training with a learning rate of {} for {} epochs with batchs of size {}".format(learning_rate,epoch,batch_size))


# In[8]:


#prep for training
net=network(learning_rate)
model_dir="model_{}_{}_{}".format(learning_rate,batch_size,epoch)
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
    
data_all=np.load("all_data.npy")  
label_all=np.load("all_labels.npy")


# In[ ]:


# train network
loss_all=[]
acc_all=[]
for e in range(epoch):
    start_time=time.time()
    perm=np.random.permutation(N)
    data_all=data_all[perm]
    label_all=label_all[perm]
    split_size=math.ceil(N/batch_size)
    data_batchs=np.array_split(data_all,split_size)
    label_batchs=np.array_split(label_all,split_size)
    loss_epoch=[]
    acc_epoch=[]

    batch_number=0
    for data,label in zip(data_batchs,label_batchs):
        batch_info=sess.run([net.mean_loss,net.mean_accuracy,net.optimizing_op],feed_dict={net.X:data, net.Y: label})
        loss_batch=batch_info[0]
        acc_batch=batch_info[1]
        loss_epoch.append(loss_batch)
        acc_epoch.append(acc_batch)
        batch_number+=1
        if batch_number%10==0:
            print("epoch_{:03d}-batch_{:03d}: loss={}, acc={}".format(e,batch_number,loss_batch,acc_batch))
    loss_all.append(loss_epoch)
    acc_all.append(acc_epoch)
    net.save("{}/epoch_{:03d}".format(train_dir,e),loss_all,acc_all)
    print("epoch took %d seconds"%(time.time()-start_time))

