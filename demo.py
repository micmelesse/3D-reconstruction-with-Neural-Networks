
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import sys
import math
import utils
import random
import layers
import dataset
import binvox_rw
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf

# tf.contrib.eager.enable_eager_execution() 
# load data
x=tf.placeholder(tf.float32,[36,24,137,137,3]) 
y=tf.placeholder(tf.float32,[36,32,32,32])

shapenet=dataset.ShapeNet()
# shapenet.batch_size=1
# data,label=shapenet.next_train_batch()


# In[3]:


# encoder network
cur_tensor=x
encoder_outputs=[cur_tensor]
print(cur_tensor.shape)
k_s = [3,3]
conv_filter_count = [96, 128, 256, 256, 256, 256]
for i in range(6): 
    ks=[7,7]if i is 0 else k_s  
    # with tf.name_scope("encoding_block"):
    
    cur_tensor=tf.map_fn(lambda x:tf.layers.conv2d(x,filters=conv_filter_count[i],padding='SAME',kernel_size= k_s,activation=None),cur_tensor)
    cur_tensor=tf.map_fn(lambda x:tf.layers.max_pooling2d(x,2,2),cur_tensor)
    cur_tensor=tf.map_fn(tf.nn.relu,cur_tensor)
    print(cur_tensor.shape)
    encoder_outputs.append(cur_tensor)

# flatten tensors
cur_tensor=tf.map_fn(tf.contrib.layers.flatten,cur_tensor)
cur_tensor=tf.map_fn(lambda x:tf.contrib.layers.fully_connected(x,1024,activation_fn=None),cur_tensor)
encoder_outputs.append(cur_tensor)
print(cur_tensor.shape)


# In[4]:


# recurrent module
cur_tensor=encoder_outputs[-1]
stacked_input=cur_tensor
for i in range(3):
    stacked_input=tf.stack([stacked_input]*4,axis=0)
print(stacked_input.shape)
    
# grid takes batches of sequences
lstm3D=layers.lstm_grid();
cur_state=lstm3D.state
for t in range(24):
    batch_frames=stacked_input[:,:,:,:,t,:]
    cur_tensor, cur_state=lstm3D.lstm_grid_call(batch_frames, cur_state)

print(cur_tensor.shape)
cur_tensor=tf.transpose(cur_tensor,[3,0,1,2,4])
print(cur_tensor.shape)


# In[5]:


# decoding network
# batch_size=cur_tensor.shape.as_list()[0]
# cur_tensor=tf.reshape(cur_tensor,[batch_size,2,2,2,-1])
# print(cur_tensor.shape)

decoder_outputs=[]
cur_tensor=layers.unpool3D(cur_tensor)
print(cur_tensor.shape)
decoder_outputs.append(cur_tensor)

k_s = [3,3,3]
deconv_filter_count = [128, 128, 128, 64, 32, 2]
for i in range(2,4): 
    # with tf.name_scope("decoding_block"):
    cur_tensor=tf.layers.conv3d_transpose(cur_tensor,padding='SAME',filters=deconv_filter_count[i],kernel_size= k_s,activation=None)
    cur_tensor=layers.unpool3D(cur_tensor)
    cur_tensor=tf.nn.relu(cur_tensor)
    print(cur_tensor.shape)
    decoder_outputs.append(cur_tensor)
            
for i in range(4,6): 
    # with tf.name_scope("decoding_block_without_unpooling"):
    cur_tensor=tf.layers.conv3d_transpose(cur_tensor,padding='SAME',filters=deconv_filter_count[i],kernel_size= k_s,activation=None)
    cur_tensor=tf.nn.relu(cur_tensor)
    print(cur_tensor.shape)
    decoder_outputs.append(cur_tensor)


# In[6]:


#3d voxel-wise softmax
y_hat=tf.nn.softmax(decoder_outputs[-1])
p=y_hat[:,:,:,:,0]
q=y_hat[:,:,:,:,1]
cross_entropies=tf.reduce_sum(-tf.multiply(tf.log(p),y)-tf.multiply(tf.log(q),1-y),[1,2,3])
loss=tf.reduce_mean(cross_entropies)
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


# In[ ]:


# setup training
sess=tf.Session()
root_train_dir = "train_dir"
cur_time = str(datetime.now().strftime('%I:%M:%S%p %Y/%m/%d'))
train_dir=os.path.join(root_train_dir,'session_{}'.format(cur_time))
saver = tf.train.Saver()
init=tf.global_variables_initializer()
sess.run(init)

# train network
print("starting training at {}".format(cur_time))
loss_session=[]
loss_all=[]
epoch=5
for e in range(epoch):
    loss_epoch=[]
    # print("starting epoch_{:03d}".format(e))
    epoch_dir="{}/epoch_{:03d}".format(train_dir,e)
    os.makedirs(epoch_dir)
    batch_number=0
    shapenet.reset()
    data,label=shapenet.next_train_batch()
    while(data is not None): 
        fd={x:data, y: label};
        batch_info=sess.run([loss,tf.trainable_variables()],feed_dict=fd)
        loss_batch=batch_info[0]
#         weight=batch_info[1][0]
#         print(type(weight))
#         np.savetxt("weight.txt",weight)
        loss_epoch.append(loss_batch) 
        train=shapenet.next_train_batch() # update
        batch_number+=1
        # show info about current batch
        if batch_number%100==0:
            print("epoch_{:03d}-batch_{:03d}: loss={}".format(e,batch_number,loss_batch))

    loss_session.append(loss_epoch)
    loss_all+=loss_epoch
    # record parameters and generate plots 
    fig = plt.figure()
    plt.plot(loss_session)
    plt.savefig("{}/loss.png".format(epoch_dir),bbox_inches='tight')
    saver.save(sess,"{}/model.ckpt".format(epoch_dir))
    plt.close()
    # save epoch losses


# In[ ]:


# tensorboard
writer = tf.summary.FileWriter("./logs/")
writer.add_graph(sess.graph)


# In[ ]:


im_3d=encoder_outputs[1].eval(session=sess,feed_dict=fd)[0]
im_2d=utils.flatten_multichannel_image(im_3d)
plt.imsave("test.png",im_2d)


# In[ ]:


# model output as a voxelized image
out=tf.cast(tf.argmax(y_hat,axis=4),dtype=tf.float32)
out=sess.run(out,feed_dict=feed_dict)
outvoxel=binvox_rw.Voxels(out,out.shape,[0,0,0],1,'xzy')
with open("out/cur_output.binvox",'w') as f:
    outvoxel.write(f)

