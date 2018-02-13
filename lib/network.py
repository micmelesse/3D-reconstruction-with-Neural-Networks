import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import lib.recurrent_module as recurrent_module
import lib.utils as utils
import lib.params as params
from datetime import datetime

# Recurrent Reconstruction Neural Network (R2N2)


class R2N2:
    def __init__(self):
        self.create_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        # read params
        with open("config/train.params") as f:
            self.learn_rate = float(params.read_param(f.readline()))
            self.batch_size = int(params.read_param(f.readline()))
            self.epoch = int(params.read_param(f.readline()))

            print("learn_rate {}, epochs {}, batch_size {}".format(
                self.learn_rate, self.epoch, self.batch_size))

        # place holders
        print("creating network...")
        self.X = tf.placeholder(tf.uint8, [None, 24, 137, 137, 4])
        self.Y = tf.placeholder(tf.uint8, [None, 32, 32, 32])

        print("encoder_network")
        with tf.name_scope("encoder_network"):
            cur_tensor = tf.cast(self.X, tf.float32)
           # self.encoder_outputs = [cur_tensor]
            k_s = [3, 3]
            conv_filter_count = [96, 128, 256, 256, 256, 256]
            for i in range(6):
                k_s = [7, 7] if i is 0 else k_s
                cur_tensor = tf.map_fn(lambda a: tf.layers.conv2d(
                    a, filters=conv_filter_count[i], padding='SAME', kernel_size=k_s, activation=None),  cur_tensor)
                cur_tensor = tf.map_fn(
                    lambda a: tf.layers.max_pooling2d(a, 2, 2),  cur_tensor)
                cur_tensor = tf.map_fn(tf.nn.relu,  cur_tensor)
                # self.encoder_outputs.append(cur_tensor)

            cur_tensor = tf.map_fn(tf.contrib.layers.flatten,  cur_tensor)
            cur_tensor = tf.map_fn(lambda a: tf.contrib.layers.fully_connected(
                a, 1024, activation_fn=None), cur_tensor)
            # self.encoder_outputs.append(cur_tensor)

        print("recurrent_module")
        with tf.name_scope("recurrent_module"):
            N, n_x, n_h = 4, 1024, 256
            self.recurrent_module = recurrent_module.GRU_GRID(
                n_cells=N, n_input=n_x, n_hidden_state=n_h)

            # self.hidden_state_list = []
            hidden_state = tf.zeros([1, 4, 4, 4, 256])
            # self.hidden_state_list.append(hidden_state)
            for t in range(24):  # feed batches of seqeuences
                hidden_state = self.recurrent_module.call(
                    cur_tensor[:, t, :], hidden_state)
                # self.hidden_state_list.append(hidden_state)
        cur_tensor = hidden_state

        print("decoder_network")
        with tf.name_scope("decoder_network"):
            # self.decoder_outputs = [cur_tensor]
            cur_tensor = utils.r2n2_unpool3D(cur_tensor)
            # self.decoder_outputs.append(cur_tensor)

            k_s = [3, 3, 3]
            deconv_filter_count = [128, 128, 128, 64, 32, 2]
            for i in range(2, 4):
                cur_tensor = tf.layers.conv3d(
                    cur_tensor, padding='SAME', filters=deconv_filter_count[i], kernel_size=k_s, activation=None)
                cur_tensor = utils.r2n2_unpool3D(cur_tensor)
                cur_tensor = tf.nn.relu(cur_tensor)
                # self.decoder_outputs.append(cur_tensor)

            for i in range(4, 6):
                cur_tensor = tf.layers.conv3d(
                    cur_tensor, padding='SAME', filters=deconv_filter_count[i], kernel_size=k_s, activation=None)
                cur_tensor = tf.nn.relu(cur_tensor)
                # self.decoder_outputs.append(cur_tensor)

        print("loss_function")
        self.logits = cur_tensor
        self.softmax = tf.nn.softmax(self.logits)
        self.log_softmax = tf.nn.log_softmax(self.logits)  # avoids log(0)
        self.prediction = tf.argmax(self.softmax, -1)
        self.label = tf.cast(tf.one_hot(self.Y, 2), self.log_softmax.dtype)
        self.cross_entropy = tf.reduce_sum(-tf.multiply(self.label,
                                                        self.log_softmax), axis=-1)

        self.losses = tf.reduce_mean(self.cross_entropy, axis=[1, 2, 3])
        self.batch_loss = tf.reduce_mean(self.losses)
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.inverse_time_decay(
            self.learn_rate, self.global_step, 10.0, 0.5)
        self.optimizing_op = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate).minimize(self.batch_loss, global_step=self.global_step)

        print("...network created")
        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

    def restore(self, model_dir):
        self.saver = tf.train.import_meta_graph(
            "{}/model.ckpt.meta".format(model_dir))
        self.saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))

    def save(self, save_dir, arr_name, vals):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        np.save("{}/{}".format(save_dir, arr_name), np.array(vals))
        self.saver.save(self.sess, "{}/model.ckpt".format(save_dir))
        self.plot(save_dir, arr_name, vals)

    def predict(self, x):
        return self.sess.run([self.prediction], {self.X: x})[0]

    def plot(self, plot_dir, plot_name, vals):
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)

        plt.plot((np.array(vals)).flatten())
        plt.savefig("{}/{}.png".format(plot_dir, plot_name),
                    bbox_inches='tight')
        plt.close()

    def train_step(self, data, label):
        x = utils.to_npy(data)
        y = utils.to_npy(label)
        return self.sess.run([self.batch_loss, self.optimizing_op], {self.X: x, self.Y: y})[0]

    def vis(self, log_dir="./log"):
        writer = tf.summary.FileWriter(log_dir)
        writer.add_graph(self.sess.graph)

    def get_encoder_state(self, x, y, state_dir="./out/state/"):
        if not os.path.isdir(state_dir):
            os.makedirs(state_dir)
        fd = {self.X: x, self.Y: y}
        states_all = []

        n_encoder = len(self.encoder_outputs)
        for l in range(n_encoder):
            state = self.encoder_outputs[l].eval(fd)
            states_all.append(state)

        return states_all

    def get_hidden_state(self, x, y, state_dir="./out/state/"):
        if not os.path.isdir(state_dir):
            os.makedirs(state_dir)
        fd = {self.X: x, self.Y: y}
        states_all = []

        n_hidden = len(self.hidden_state_list)
        for l in range(n_hidden):
            state = self.hidden_state_list[l].eval(fd)
            states_all.append(state)

        return states_all

    def get_decoder_state(self, x, y, state_dir="./out/state/"):
        if not os.path.isdir(state_dir):
            os.makedirs(state_dir)
        fd = {self.X: x, self.Y: y}
        states_all = []

        n_decoder = len(self.decoder_outputs)
        for l in range(n_decoder):
            state = self.decoder_outputs[l].eval(fd)
            states_all.append(state)

        states_all.append(self.softmax.eval(fd))
        return states_all
