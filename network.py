import utils
import tensorflow as tf



""" encoder network """


def encoder(value):
    cur_tensor = value
    # print(cur_tensor.shape)
    k_s = [3, 3]
    conv_filter_count = [96, 128, 256, 256, 256, 256]
    for i in range(6):
        ks = [7, 7]if i is 0 else k_s
        with tf.name_scope("encoding_block"):
            cur_tensor = tf.layers.conv2d(
                cur_tensor, filters=conv_filter_count[i], padding='SAME', kernel_size=k_s, activation=None)
            cur_tensor = tf.layers.max_pooling2d(cur_tensor, 2, 2)
            cur_tensor = tf.nn.relu(cur_tensor)
            # print(cur_tensor.shape)

    # flatten tensor
    cur_tensor = tf.contrib.layers.flatten(cur_tensor)
    cur_tensor = tf.contrib.layers.fully_connected(
        cur_tensor, 1024, activation_fn=None)
    # print(cur_tensor.shape)

    return cur_tensor


""" decoding network """


def decoder(value):

    cur_tensor = value
    k_s = [3, 3, 3]
    deconv_filter_count = [128, 128, 128, 64, 32, 2]
    cur_tensor = utils.unpool(cur_tensor)
    # print(cur_tensor.shape)
    for i in range(2, 4):
        with tf.name_scope("decoding_block"):
                cur_tensor = tf.layers.conv3d_transpose(
                    cur_tensor, padding='SAME', filters=deconv_filter_count[i], kernel_size=k_s, activation=None)
                cur_tensor = utils.unpool(cur_tensor)
                cur_tensor = tf.nn.relu(cur_tensor)
                # print(cur_tensor.shape)

    for i in range(4, 6):
        with tf.name_scope("decoding_block_without_unpooling"):
                cur_tensor = tf.layers.conv3d_transpose(
                    cur_tensor, padding='SAME', filters=deconv_filter_count[i], kernel_size=k_s, activation=None)
                cur_tensor = tf.nn.relu(cur_tensor)
                # print(cur_tensor.shape)

    return cur_tensor


""" 3D_Convolutional_LSTM """

def recurrent_module(value):
    cur_tensor=value
    with tf.name_scope("3D_LSTM"):
        # x=tf.placeholder(tf.float32,[None,1024])
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
    return cur_tensor

