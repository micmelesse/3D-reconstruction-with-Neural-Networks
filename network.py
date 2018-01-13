import tensorflow as tf


class network:
    def __init__(self):
        # place holders
        self.x = tf.placeholder(tf.float32, [None, 24, 137, 137, 3])
        self.y = tf.placeholder(tf.float32, [None, 32, 32, 32])

        # encoder network
        cur_tensor = self.x
        self.encoder_outputs = [cur_tensor]
        print(cur_tensor.shape)
        k_s = [3, 3]
        conv_filter_count = [96, 128, 256, 256, 256, 256]
        for i in range(6):
            ks = [7, 7]if i is 0 else k_s
            with tf.name_scope("encoding_block"):
                 cur_tensor = tf.map_fn(lambda a: tf.layers.conv2d(
                    a, filters=conv_filter_count[i], padding='SAME', kernel_size=k_s, activation=None),  cur_tensor)
                 cur_tensor = tf.map_fn(
                    lambda a: tf.layers.max_pooling2d(a, 2, 2),  cur_tensor)
                 cur_tensor = tf.map_fn(tf.nn.relu,  cur_tensor)
                print(cur_tensor.shape)
                 self.encoder_outputs.append( cur_tensor)

        # flatten tensors
         cur_tensor = tf.map_fn(tf.contrib.layers.flatten,  cur_tensor)
         cur_tensor = tf.map_fn(lambda a: tf.contrib.layers.fully_connected(
            a, 1024, activation_fn=None), cur_tensor)
         self.encoder_outputs.append(cur_tensor)
        print( cur_tensor.shape)
        
        # recurrent module
        with tf.name_scope("recurrent_module"): 
            
            N,n_x,n_h=4,1024,256
        
            self.recurrent_module=layers.GRU_R2N2(n_cells=N,n_input=n_x,n_hidden_state=n_h)

            # initial hidden state
            self.hidden_state_list=[]
            with open("net.params") as f:
                batch_size=int(f.readline())
                hidden_state= tf.zeros([1,4,4,4,256])
            
            # feed batches of seqeuences
            for t in range(24):
                fc_batch_t = cur_tensor[:,t,:]
                self.hidden_state_list.append(hidden_state)
                hidden_state = self.recurrent_module.call(fc_batch_t, hidden_state)
            print(hidden_state.shape)
        cur_tensor=hidden_state

                # decoding network
        self.decoder_outputs=[cur_tensor]
        cur_tensor=layers.unpool3D(cur_tensor)
        print(cur_tensor.shape)
        self.decoder_outputs.append(cur_tensor)

        k_s = [3,3,3]
        deconv_filter_count = [128, 128, 128, 64, 32, 2]
        for i in range(2,4): 
            with tf.name_scope("decoding_block"):
                cur_tensor=tf.layers.conv3d(cur_tensor,padding='SAME',filters=deconv_filter_count[i],kernel_size= k_s,activation=None)
                cur_tensor=layers.unpool3D(cur_tensor)
                cur_tensor=tf.nn.relu(cur_tensor)
                print(cur_tensor.shape)
                self.decoder_outputs.append(cur_tensor)
                    
        for i in range(4,6): 
            with tf.name_scope("decoding_block_without_unpooling"):
                cur_tensor=tf.layers.conv3d(cur_tensor,padding='SAME',filters=deconv_filter_count[i],kernel_size= k_s,activation=None)
                cur_tensor=tf.nn.relu(cur_tensor)
                print(cur_tensor.shape)
                self.decoder_outputs.append(cur_tensor)

        with tf.name_scope("cost"):
            # 3d voxel-wise softmax
            y_hat=tf.nn.softmax(decoder_outputs[-1])
            p,q=y_hat[:,:,:,:,0],y_hat[:,:,:,:,1]
            self.cross_entropies=tf.reduce_sum(-tf.multiply(tf.log(p),y)-tf.multiply(tf.log(q),1-y),[1,2,3])
            self.optimizing_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropies)

            self.prediction=tf.to_float(tf.argmax(y_hat,axis=4))
            accuracies=tf.reduce_sum(tf.to_float(tf.equal(y,prediction)),axis=[1,2,3])/tf.constant(32*32*32,dtype=tf.float32)
            self.mean_loss=tf.reduce_mean(cross_entropies)
            self.mean_accuracy=tf.reduce_mean(accuracies)

