
# coding: utf-8

# In[ ]:


import os
import sys
import utils
import random
import dataset
import binvox_rw
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

#import tensorflow.contrib.eager as tfe
#tfe.enable_eager_execution()
sess=tf.InteractiveSession()
# load data
shapenet=dataset.ShapeNet()
#shapenet.batch_size=3


# In[ ]:


with tf.name_scope('input'):
    x=tf.placeholder(tf.float32,[shapenet.batch_size,137,137,3])
    y=tf.placeholder(tf.float32,[shapenet.batch_size,32,32,32])


# In[ ]:


# encoder network
cur_tensor=x
print(cur_tensor.shape)
k_s = [3,3]
conv_filter_count = [96, 128, 256, 256, 256, 256]
for i in range(6): 
    ks=[7,7]if i is 0 else k_s  
    with tf.name_scope("encoding_block"):
        cur_tensor=tf.layers.conv2d(cur_tensor,filters=conv_filter_count[i],padding='SAME',kernel_size= k_s,activation=None)
        cur_tensor=tf.layers.max_pooling2d(cur_tensor,2,2)
        cur_tensor=tf.nn.relu(cur_tensor)
        print(cur_tensor.shape)

# flatten tensor
cur_tensor=tf.contrib.layers.flatten(cur_tensor)
cur_tensor=tf.contrib.layers.fully_connected(cur_tensor,1024,activation_fn=None)
print(cur_tensor.shape)


# In[ ]:


# 3D_Convolutional_LSTM 

with tf.name_scope("3D_LSTM"):
    #x=tf.placeholder(tf.float32,[None,1024])
    h_t=tf.Variable(tf.zeros([4,4,4,1,256]),name="hidden_state")
    s_t=tf.Variable(tf.zeros([4,4,4,1,256]),name="cell_state")

    # W,U,b for f,i,s,o
    W=[tf.Variable(tf.zeros([1024,256]),name="W%d"%(i)) for i in range(4)]
    U=[tf.Variable(tf.zeros([3,3,3,256,256]),name="U%d"%(i)) for i in range(4)]
    b=[tf.Variable(tf.zeros([1,256]),name="b%d"%(i)) for i in range(4)]

    Wx=[tf.matmul(cur_tensor, W[i]) for i in range(4)]
    Uh=[tf.nn.convolution(h_t,U[i],padding="SAME") for i in range(4)]


    f_t=tf.sigmoid(Wx[0] + Uh[0] + b[0])
    i_t=tf.sigmoid(Wx[1] + Uh[1] + b[1])
    o_t=tf.sigmoid(Wx[2]+ Uh[2] + b[2])
    s_t=tf.multiply(f_t,s_t) + tf.multiply(i_t,tf.tanh(Wx[3]+ Uh[3] + b[3]))
    h_t=tf.tanh(s_t)

    cur_tensor=tf.transpose(h_t,[3,0,1,2,4])
    print(cur_tensor.shape)


# In[ ]:


# decoding network
# batch_size=cur_tensor.shape.as_list()[0]
# cur_tensor=tf.reshape(cur_tensor,[batch_size,2,2,2,-1])
# print(cur_tensor.shape)
k_s = [3,3,3]
deconv_filter_count = [128, 128, 128, 64, 32, 2]
cur_tensor=utils.unpool(cur_tensor)
print(cur_tensor.shape)
for i in range(2,4): 
    with tf.name_scope("decoding_block"):
            cur_tensor=tf.layers.conv3d_transpose(cur_tensor,padding='SAME',filters=deconv_filter_count[i],kernel_size= k_s,activation=None)
            cur_tensor=utils.unpool(cur_tensor)
            cur_tensor=tf.nn.relu(cur_tensor)
            print(cur_tensor.shape)
            
for i in range(4,6): 
    with tf.name_scope("decoding_block_without_unpooling"):
            cur_tensor=tf.layers.conv3d_transpose(cur_tensor,padding='SAME',filters=deconv_filter_count[i],kernel_size= k_s,activation=None)
            cur_tensor=tf.nn.relu(cur_tensor)
            print(cur_tensor.shape)

y_hat=cur_tensor


# In[ ]:


# tensorboard
writer = tf.summary.FileWriter("./logs/")
writer.add_graph(sess.graph)
saver = tf.train.Saver()


# train network
epoch=5
tf.global_variables_initializer().run()


#3d voxel-wise softmax
y_hat=tf.nn.softmax(y_hat)
p=y_hat[:,:,:,:,0]
q=y_hat[:,:,:,:,1]
loss_voxel=tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.log(p),y)+tf.multiply(tf.log(q),1-y),[1,2,3]))
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss_voxel)

loss=[]
for e in range(1,epoch+1):
    print("epoch {}".format(e))
    train=shapenet.next_train_batch()
    while(train is not None):
        train_X=dataset.load_dataset(train[:,0])
        train_Y=dataset.load_labels(train[:,1])
        fd={x:train_X , y: train_Y};
        l=sess.run([loss_voxel],feed_dict=fd)
        loss.append(l)  
        train=shapenet.next_train_batch()  
    loss.append(np.nan)
    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)
    fig = plt.figure()
    plt.plot(loss)
    plt.savefig("plots/{}.png".format(e),bbox_inches='tight')
    plt.close()
        
    


# In[ ]:


# model output as a voxelized image
out=tf.cast(tf.argmax(y_hat,axis=4),dtype=tf.float32)
out=sess.run(out,feed_dict=feed_dict)
outvoxel=binvox_rw.Voxels(out,out.shape,[0,0,0],1,'xzy')
with open("out/{i}.binvox",'w') as f:
    outvoxel.write(f)

