import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import lib.recurrent_module as recurrent_module
import lib.utils as utils

# Recurrent Reconstruction Neural Network (R2N2)


class R2N2:
    def __init__(self, lr):
        # place holders
        self.X = tf.placeholder(tf.uint8, [None, 24, 137, 137, 4])
        self.Y = tf.placeholder(tf.uint8, [None, 32, 32, 32])

        # print("encoder_network")
        with tf.name_scope("encoder_network"):
            self.input = cur_tensor = tf.cast(self.X, tf.float16)
            # print(cur_tensor.shape)
            self.encoder_outputs = [cur_tensor]
            k_s = [3, 3]
            conv_filter_count = [96, 128, 256, 256, 256, 256]
            for i in range(6):
                k_s = [7, 7] if i is 0 else k_s
                cur_tensor = tf.map_fn(lambda a: tf.layers.conv2d(
                    a, filters=conv_filter_count[i], padding='SAME', kernel_size=k_s, activation=None),  cur_tensor)
                cur_tensor = tf.map_fn(
                    lambda a: tf.layers.max_pooling2d(a, 2, 2),  cur_tensor)
                cur_tensor = tf.map_fn(tf.nn.relu,  cur_tensor)
                # print(cur_tensor.shape)
                self.encoder_outputs.append(cur_tensor)

            cur_tensor = tf.map_fn(tf.contrib.layers.flatten,  cur_tensor)
            cur_tensor = tf.map_fn(lambda a: tf.contrib.layers.fully_connected(
                a, 1024, activation_fn=None), cur_tensor)
            # print(cur_tensor.shape)
            self.final_encoder_state = cur_tensor
            self.encoder_outputs.append(cur_tensor)

        # print("recurrent_module")
        with tf.name_scope("recurrent_module"):
            N, n_x, n_h = 4, 1024, 256
            self.recurrent_module = recurrent_module.GRU_GRID(
                n_cells=N, n_input=n_x, n_hidden_state=n_h)

            self.hidden_state_list = []  # initial hidden state
            hidden_state = tf.zeros([1, 4, 4, 4, 256])
            self.hidden_state_list.append(hidden_state)

            for t in range(24):  # feed batches of seqeuences
                hidden_state = self.recurrent_module.call(
                    cur_tensor[:, t, :], hidden_state)
                self.hidden_state_list.append(hidden_state)
            # print(hidden_state.shape)
        self.final_hidden_state = hidden_state
        cur_tensor = hidden_state

        # print("decoder_network")
        with tf.name_scope("decoder_network"):
            self.decoder_outputs = [cur_tensor]
            cur_tensor = utils.r2n2_unpool3D(cur_tensor)
            # print(cur_tensor.shape)
            self.decoder_outputs.append(cur_tensor)

            k_s = [3, 3, 3]
            deconv_filter_count = [128, 128, 128, 64, 32, 2]
            for i in range(2, 4):
                cur_tensor = tf.layers.conv3d(
                    cur_tensor, padding='SAME', filters=deconv_filter_count[i], kernel_size=k_s, activation=None)
                cur_tensor = utils.r2n2_unpool3D(cur_tensor)
                cur_tensor = tf.nn.relu(cur_tensor)
                # print(cur_tensor.shape)
                self.decoder_outputs.append(cur_tensor)

            for i in range(4, 6):
                cur_tensor = tf.layers.conv3d(
                    cur_tensor, padding='SAME', filters=deconv_filter_count[i], kernel_size=k_s, activation=None)
                cur_tensor = tf.nn.relu(cur_tensor)
                # print(cur_tensor.shape)
                self.decoder_outputs.append(cur_tensor)

        # print("loss")
        self.logits = self.final_decoder_state = cur_tensor
        self.label = tf.one_hot(self.Y, 2)
        self.softmax = tf.nn.softmax(self.logits)
        self.log_softmax = tf.log(self.softmax)
        self.cross_entropy = tf.reduce_sum(-tf.multiply(tf.cast(self.label, self.log_softmax.dtype),
                                                        self.log_softmax), axis=-1)

        self.losses = tf.reduce_mean(self.cross_entropy, axis=[1, 2, 3])
        self.batch_loss = tf.reduce_mean(self.losses)
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.inverse_time_decay(
            lr, self.global_step, 1.0, 0.5)
        self.optimizing_op = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate).minimize(self.batch_loss, global_step=self.global_step)

        # init session
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        tf.global_variables_initializer().run()

    def plot(self, plot_dir, plot_name, vals):
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)

        plt.plot((np.array(vals)).flatten())
        plt.savefig("{}/{}.png".format(plot_dir, plot_name),
                    bbox_inches='tight')
        plt.close()

    def save(self, save_dir, arr_name, vals):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        np.save("{}/{}".format(save_dir, arr_name), np.array(vals))
        self.saver.save(self.sess, "{}/model.ckpt".format(save_dir))
        self.plot(save_dir, arr_name, vals)

    def vis(self, log_dir="./log"):  # tensorboard/ vis tools
        writer = tf.summary.FileWriter(log_dir)
        writer.add_graph(self.sess.graph)

    def train_step(self, fd):
        return self.sess.run([self.optimizing_op], fd)

    def save_encoder_state(self, save_dir, fd):
        n_layers = len(self.encoder_outputs)
        for l in range(n_layers):
            state = self.encoder_outputs[l].eval(fd)
            n_batch = state.shape[0]
            for b in range(n_batch):
                n_time = state.shape[1]
                for t in range(n_time):
                    np.save(save_dir + "/encoder_{}-{}-{}".format(l,
                                                                  b, t), state[b, t])
                    if l == n_layers - 1:
                        plt.plot(state[b, t])
                        plt.savefig(
                            save_dir + "/encoder_{}-{}-{}.png".format(l, b, t))
                        plt.close()

                    else:
                        utils.imsave_multichannel(
                            state[b, t], save_dir + "/encoder_{}-{}-{}.png".format(l, b, t))

    def save_decoder_state(self, save_dir, fd):
        n_layers = len(self.decoder_outputs)
        for l in range(n_layers):
            state = self.decoder_outputs[l].eval(fd)
            n_batch = state.shape[0]
            for b in range(n_batch):
                n_channels = state.shape[-1]
                for c in range(n_channels):
                    np.save(save_dir + "/decoder_{}-{}-{}".format(l,
                                                                  b, c), state[b, :, :, :, c])
                    utils.imsave_voxel(state[b, :, :, :, c], save_dir +
                                       "/decoder_{}-{}-{}.png".format(l, b, c))

    def save_hidden_state(self, save_dir, fd):
        n_layers = len(self.hidden_state_list)
        for l in range(n_layers):
            state = self.hidden_state_list[l].eval(fd)
            print(state.shape)
            n_batch = state.shape[0]
            for b in range(n_batch):
                n_channels = state.shape[-1]
                for c in range(n_channels):
                    np.save(save_dir + "/hidden_{}-{}-{}".format(l,
                                                                 b, c), state[b, :, :, :, c])
                    utils.imsave_voxel(state[b, :, :, :, c], save_dir +
                                       "/hidden_{}-{}-{}.png".format(l, b, c))
