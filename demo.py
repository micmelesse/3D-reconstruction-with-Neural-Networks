
# coding: utf-8

# In[ ]:


import os
import sys
import math
import utils
import random
import dataset
import binvox_rw
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()


# In[ ]:


# load data
shapenet=dataset.ShapeNet()
# shapenet.batch_size=3

# with tf.name_scope('input'):
x=tf.placeholder(tf.float32,[shapenet.batch_size,137,137,3])
y=tf.placeholder(tf.float32,[shapenet.batch_size,32,32,32])


# In[ ]:


# encoder network
cur_tensor=x
encoder_outputs=[x]
print(cur_tensor.shape)

k_s = [3,3]
conv_filter_count = [96, 128, 256, 256, 256, 256]
for i in range(6): 
    ks=[7,7]if i is 0 else k_s  
    # with tf.name_scope("encoding_block"):
    cur_tensor=tf.layers.conv2d(cur_tensor,filters=conv_filter_count[i],padding='SAME',kernel_size= k_s,activation=None)
    cur_tensor=tf.layers.max_pooling2d(cur_tensor,2,2)
    cur_tensor=tf.nn.relu(cur_tensor)
    print(cur_tensor.shape)
    encoder_outputs.append(cur_tensor)

# flatten tensor
cur_tensor=tf.contrib.layers.flatten(cur_tensor)
cur_tensor=tf.contrib.layers.fully_connected(cur_tensor,1024,activation_fn=None)
encoder_outputs.append(cur_tensor)
print(cur_tensor.shape)


# In[ ]:


# create 3D_Convolutional_LSTM 
class LSTM3D_GRID:
    def __init__(self,batch_size=24,grid_size=4,input_size=1024,hidden_state_size=256,kernel_size=3):  
        self.N=grid_size
        self.n_x=input_size
        self.n_h=hidden_state_size
        self.n_k=kernel_size
        self.batch_size=batch_size
        
        # hidden state and memory cell state
        self.hidden_state=tf.ones([batch_size,self.N,self.N,self.N,self.n_h])
        self.prev_state=tf.atanh(self.hidden_state)

        
        # the weights are shared by all the units for a specific gate
        self.W_f=tf.Variable(tf.ones([self.n_x,self.n_h]),name="W_f")
        self.W_s=tf.Variable(tf.ones([self.n_x,self.n_h]),name="W_s")
        self.W_i=tf.Variable(tf.ones([self.n_x,self.n_h]),name="W_i")

        # the kernel is just a rank 3 tensor of weights
        self.U_f=tf.Variable(tf.ones([self.n_k,self.n_k,self.n_k,256,256]),name="U_f") # each weight in the kernel weights a hidden state
        self.U_s=tf.Variable(tf.ones([self.n_k,self.n_k,self.n_k,256,256]),name="U_s")
        self.U_i=tf.Variable(tf.ones([self.n_k,self.n_k,self.n_k,256,256]),name="U_i")
        
        # biases
        self.b_f=tf.Variable(tf.ones([self.n_h]),name="b_f")
        self.b_s=tf.Variable(tf.ones([self.n_h]),name="b_s")
        self.b_i=tf.Variable(tf.ones([self.n_h]),name="b_i")
    def call(self,x):
        def gate(x,W,U,b):
            Wx=tf.matmul(x,W)
            Uh=tf.nn.conv3d(self.hidden_state,U,strides=[1,1,1,1,1],padding="SAME")
            for i in range(3): # repeatdly stack elements
                Wx=tf.stack([Wx]*4,axis=1)
            return tf.sigmoid(Wx+Uh+b)
        
        f_t=gate(x,self.W_f,self.U_f,self.b_f)
        i_t=gate(x,self.W_i,self.U_i,self.b_i)
        s_t=tf.multiply(f_t,self.prev_state)+tf.multiply(i_t,gate(x,self.W_s,self.U_s,self.b_s))
        
        # change state based on input
        self.prev_state=s_t
        self.hidden_state=tf.tanh(s_t)
        return self.hidden_state


# In[ ]:


recurrence_module=LSTM3D_GRID()
cur_tensor=encoder_outputs[-1]
cur_tensor=recurrence_module.call(cur_tensor)
print(cur_tensor.shape)


# In[ ]:


# decoding network
# batch_size=cur_tensor.shape.as_list()[0]
# cur_tensor=tf.reshape(cur_tensor,[batch_size,2,2,2,-1])
# print(cur_tensor.shape)

decoder_outputs=[]
cur_tensor=utils.unpool(cur_tensor)
print(cur_tensor.shape)
decoder_outputs.append(cur_tensor)

k_s = [3,3,3]
deconv_filter_count = [128, 128, 128, 64, 32, 2]
for i in range(2,4): 
    # with tf.name_scope("decoding_block"):
    cur_tensor=tf.layers.conv3d_transpose(cur_tensor,padding='SAME',filters=deconv_filter_count[i],kernel_size= k_s,activation=None)
    cur_tensor=utils.unpool(cur_tensor)
    cur_tensor=tf.nn.relu(cur_tensor)
    print(cur_tensor.shape)
    decoder_outputs.append(cur_tensor)
            
for i in range(4,6): 
    # with tf.name_scope("decoding_block_without_unpooling"):
    cur_tensor=tf.layers.conv3d_transpose(cur_tensor,padding='SAME',filters=deconv_filter_count[i],kernel_size= k_s,activation=None)
    cur_tensor=tf.nn.relu(cur_tensor)
    print(cur_tensor.shape)
    decoder_outputs.append(cur_tensor)


# In[ ]:


#3d voxel-wise softmax
y_hat=tf.nn.softmax(decoder_outputs[-1])
p=y_hat[:,:,:,:,0]
q=y_hat[:,:,:,:,1]
cross_entropies=tf.reduce_sum(-tf.multiply(tf.log(p),y)-tf.multiply(tf.log(q),1-y),[1,2,3])
loss_voxel=tf.reduce_mean(cross_entropies)
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss_voxel)

sess=tf.Session()


# In[ ]:


# setup training
print("starting training")
root_train_dir = "train_dir"
cur_time = str(datetime.now()).translate({ord(" "): "_"})
#sys.stdout = open(cur_time+'.txt', 'w')
train_dir=os.path.join(root_train_dir,cur_time)
saver = tf.train.Saver()
init=tf.global_variables_initializer()
sess.run(init)

# train network
loss=[math.inf]
epoch=5
for e in range(epoch):     
    batch_number=0
    train=shapenet.next_train_batch()
    while(train is not None):
        iter_dir="{}/epoch_{:03d}/batch_{:03d}".format(train_dir,e,batch_number)    
        os.makedirs(iter_dir)
        print("{}: loss={}".format(iter_dir,loss[-1]))
        
        train_X=dataset.load_dataset(train[:,0])
        train_Y=dataset.load_labels(train[:,1])
        fd={x:train_X, y: train_Y};
        l=sess.run([loss_voxel],feed_dict=fd)
        loss.append(l[0]) 
        
        
        # record parameters and generate plots
        fig = plt.figure()
        plt.plot(loss)
        plt.savefig("{}/loss.png".format(iter_dir),bbox_inches='tight')
        saver.save(sess,"{}/model.ckpt".format(iter_dir))
        plt.close()
        # update
        train=shapenet.next_train_batch()
        batch_number+=1


# In[ ]:


# tensorboard
writer = tf.summary.FileWriter("./logs/")
writer.add_graph(sess.graph)


# %matplotlib inline
# %load_ext autoreload
# %autoreload 
# 
# train=shapenet.next_train_batch()
# train_X=dataset.load_dataset(train[:,0])
# train_Y=dataset.load_labels(train[:,1])
# fd={x:train_X, y: train_Y};
# 
# im_3d=encoder_outputs[1].eval(session=sess,feed_dict=fd)[0]
# im_2d=utils.flatten_multichannel_image(im_3d)
# plt.imsave("test.png",im_2d)

# # model output as a voxelized image
# out=tf.cast(tf.argmax(y_hat,axis=4),dtype=tf.float32)
# out=sess.run(out,feed_dict=feed_dict)
# outvoxel=binvox_rw.Voxels(out,out.shape,[0,0,0],1,'xzy')
# with open("out/{i}.binvox",'w') as f:
#     outvoxel.write(f)
