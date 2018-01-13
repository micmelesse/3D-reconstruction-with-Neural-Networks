
# coding: utf-8

# In[1]:


import os
import sys
import math
import utils
import random
import layers
import dataset
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from network import network


# In[2]:


#read params
with open("train.params") as f:
    batch_size=int(f.readline())
    epoch=int(f.readline())
    learning_rate=float(f.readline())
    print("training with a learning rate of {} for {} epochs with batchs of size {}".format(learning_rate,epoch,batch_size))


# In[3]:


net=network(learning_rate)


# In[4]:


# output
sess=tf.InteractiveSession()
tf.global_variables_initializer().run()

# tensorboard/ vis tools
writer = tf.summary.FileWriter("./logs/")
writer.add_graph(sess.graph)


# In[5]:


data_all=np.load("all_data.npy")  
label_all=np.load("all_labels.npy")
N=len(data_all)
print(data_all.shape)
print(label_all.shape)


# In[ ]:


# setup training
root_train_dir = "train_dir"
cur_time = str(datetime.now().strftime('%Y|%m|%d %I:%M:%S %p'))
train_dir=os.path.join(root_train_dir,cur_time)
saver = tf.train.Saver()

# train network
print("starting training at {}".format(cur_time))
loss_session=[]
acc_session=[]
for e in range(epoch):
    perm=np.random.permutation(N)
    data_all=data_all[perm]
    label_all=label_all[perm]
    split_size=math.ceil(N/batch_size)
    data_batchs=np.array_split(data_all,split_size)
    label_batchs=np.array_split(label_all,split_size)
    loss_epoch=[]
    acc_epoch=[]
    # print("starting epoch_{:03d}".format(e))
    epoch_dir="{}/epoch_{:03d}".format(train_dir,e)
    os.makedirs(epoch_dir)
    
    batch_number=0
    for data,label in zip(data_batchs,label_batchs):
        fd={net.X:data, net.Y: label};
        batch_info=sess.run([net.mean_loss,net.mean_accuracy,net.optimizing_op],feed_dict=fd)
        loss_batch=batch_info[0]
        acc_batch=batch_info[1]
        loss_epoch.append(loss_batch)
        acc_epoch.append(acc_batch)
        batch_number+=1
        if batch_number%batch_size==0:
            print("epoch_{:03d}-batch_{:03d}: loss={}, acc={}".format(e,batch_number,loss_batch,acc_batch))
    
    print("saving checkpoint for epoch {}".format(e))
    saver.save(sess,"{}/model.ckpt".format(epoch_dir))   
    loss_session.append(loss_epoch)
    acc_session.append(acc_epoch)
    np.save("{}/losses".format(epoch_dir),np.array(loss_session))
    np.save("{}/accs".format(epoch_dir),np.array(acc_session))
    fig = plt.figure()
    plt.plot((np.array(loss_session)).flatten())
    plt.savefig("{}/epoch_loss.png".format(epoch_dir),bbox_inches='tight')
    plt.close()
    fig = plt.figure()
    plt.plot((np.array(acc_session)).flatten())
    plt.ylim(0,1)
    plt.savefig("{}/epoch_acc.png".format(epoch_dir),bbox_inches='tight')
    plt.close()

