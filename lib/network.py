import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import lib.utils as utils
from datetime import datetime

# Recurrent Reconstruction Neural Network (R2N2)

N_PARALLEL = 1


class R2N2:
    def __init__(self, params=None):
        self.session_loss = []

        self.create_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        # read params
        if params is None:
            self.learn_rate, self.batch_size, self.epoch_count = utils.get_params_from_disk()
        else:
            self.learn_rate = params['learn_rate']
            self.batch_size = params['batch_size']
            self.epoch_count = params['epoch_count']

        print("learn_rate {}, epoch_count {}, batch_size {}".format(
            self.learn_rate, self.epoch_count, self.batch_size))

        # place holders
        print("creating network...")
        self.X = tf.placeholder(tf.float32, [None, 24, 137, 137, 4])
        self.Y = tf.placeholder(tf.uint8, [None, 32, 32, 32])
        cur_tensor = self.X

        print("encoder_network")
        with tf.name_scope("encoder_network"):
            k_s = [3, 3]
            conv_filter_count = [96, 128, 256, 256, 256, 256]

            for i in range(7):
                if i < 6:
                    k_s = [7, 7] if i is 0 else k_s
                    cur_tensor = tf.map_fn(lambda a: tf.layers.conv2d(
                        a, filters=conv_filter_count[i], padding='SAME', kernel_size=k_s, activation=None),  cur_tensor, parallel_iterations=N_PARALLEL)
                    cur_tensor = tf.map_fn(
                        lambda a: tf.layers.max_pooling2d(a, 2, 2),  cur_tensor, parallel_iterations=N_PARALLEL)
                    cur_tensor = tf.map_fn(
                        tf.nn.relu,  cur_tensor, parallel_iterations=N_PARALLEL)
                elif i == 6:
                    cur_tensor = tf.map_fn(
                        tf.contrib.layers.flatten,  cur_tensor, parallel_iterations=N_PARALLEL)
                    cur_tensor = tf.map_fn(lambda a: tf.contrib.layers.fully_connected(
                        a, 1024, activation_fn=None), cur_tensor, parallel_iterations=N_PARALLEL)
                    cur_tensor = tf.map_fn(
                        tf.nn.relu,  cur_tensor, parallel_iterations=N_PARALLEL)
                # print(cur_tensor.shape)

        cur_tensor = tf.verify_tensor_all_finite(
            cur_tensor, "fc vector (encoder output)")

        print("recurrent_module")
        with tf.name_scope("recurrent_module"):
            rnn = GRU_GRID()
            hidden_state = None
            for t in range(24):  # feed batches of seqeuences
                hidden_state = tf.verify_tensor_all_finite(rnn.call(
                    cur_tensor[:, t, :], hidden_state), "hidden_state {}".format(t))
        cur_tensor = hidden_state
        # print(cur_tensor.shape)

        print("decoder_network")
        with tf.name_scope("decoder_network"):
            k_s = [3, 3, 3]
            deconv_filter_count = [128, 128, 128, 64, 32, 2]

            for i in range(6):
                if i == 0:
                    cur_tensor = r2n2_unpool3D(cur_tensor)
                elif i in range(1, 3):  # scale up hidden state to 32*32*32
                    cur_tensor = tf.layers.conv3d(
                        cur_tensor, padding='SAME', filters=deconv_filter_count[i], kernel_size=k_s, activation=None)
                    cur_tensor = tf.nn.relu(cur_tensor)
                    cur_tensor = r2n2_unpool3D(cur_tensor)
                elif i in range(3, 5):  # reduce number of channels to 2
                    cur_tensor = tf.layers.conv3d(
                        cur_tensor, padding='SAME', filters=deconv_filter_count[i], kernel_size=k_s, activation=None)
                    cur_tensor = tf.nn.relu(cur_tensor)
                elif i == 5:  # final conv before softmax
                    cur_tensor = tf.layers.conv3d(
                        cur_tensor, padding='SAME', filters=deconv_filter_count[i], kernel_size=k_s, activation=None)
                # print(cur_tensor.shape)

        print("loss_function")
        logits = tf.verify_tensor_all_finite(
            cur_tensor, "logits (decoder output)")
        softmax = tf.nn.softmax(logits)
        log_softmax = tf.nn.log_softmax(logits)  # avoids log(0)
        label = tf.one_hot(self.Y, 2)
        cross_entropy = tf.reduce_sum(-tf.multiply(label,
                                                   log_softmax), axis=-1)
        losses = tf.reduce_mean(cross_entropy, axis=[1, 2, 3])
        batch_loss = tf.reduce_mean(losses)
        self.loss = batch_loss

        # misc
        step_count = tf.Variable(0, trainable=False)
        lr = self.learn_rate
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=lr)
        grads_and_vars = optimizer.compute_gradients(batch_loss)
        map(lambda a: tf.verify_tensor_all_finite(
            a[0], "grads_and_vars"), grads_and_vars)  # assert no Nan or Infs in grad

        self.final_op = optimizer.apply_gradients(
            grads_and_vars, global_step=step_count)
        self.print = tf.Print(batch_loss, [batch_loss, lr])

        print("...network created")
        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        self.prediction = tf.argmax(softmax, -1)
        tf.global_variables_initializer().run()

    def train_step(self, data, label):
        x = utils.to_npy(data)
        y = utils.to_npy(label)
        return self.sess.run([self.final_op, self.print, self.loss], {self.X: x, self.Y: y})[2]

    def save(self, save_dir):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.saver.save(self.sess, "{}/model.ckpt".format(save_dir))
        np.save("{}/loss.npy".format(save_dir), self.session_loss)
        self.plot_loss(save_dir, self.session_loss)

    def restore(self, model_dir):
        self.saver = tf.train.import_meta_graph(
            "{}/model.ckpt.meta".format(model_dir))
        self.saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))

    def predict(self, x):
        return self.sess.run([self.prediction], {self.X: x})[0]

    def plot_loss(self, plot_dir, loss_arr):
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)
        plt.plot(np.array(loss_arr).flatten())
        plt.savefig("{}/loss.png".format(plot_dir), bbox_inches='tight')
        plt.close()

    def vis(self, log_dir="./log"):
        writer = tf.summary.FileWriter(log_dir)
        writer.add_graph(self.sess.graph)


def r2n2_unpool3D(value, name='unpool3D'):
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1: -1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat([out, tf.zeros_like(out)], i)
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out


class GRU_GRID:
    def __init__(self):
        self.N = 3
        self.n_cells = 4
        self.n_input = 1024
        self.n_hidden_state = 128

        self.W = tf.Variable(tf.random_uniform(
            [self.N,  self.n_cells,  self.n_cells,  self.n_cells,  self.n_input,  self.n_hidden_state]), name="W_GRU")
        self.b = tf.Variable(tf.random_uniform(
            [self.N,  self.n_cells,  self.n_cells,  self.n_cells,  self.n_hidden_state]), name="b_GRU")
        self.U = tf.Variable(tf.random_uniform(
            [self.N, 3, 3, 3,  self.n_hidden_state,  self.n_hidden_state]), name="U_GRU")

    def call(self, fc_input, prev_state):
        if prev_state is None:
            prev_state = tf.zeros(
                [1, self.n_cells, self.n_cells, self.n_cells,  self.n_hidden_state])

        fc_input = r2n2_stack(fc_input)
        u_t = tf.sigmoid(
            r2n2_linear(fc_input, self.W[0], self.U[0], prev_state, self.b[0]))
        r_t = tf.sigmoid(
            r2n2_linear(fc_input, self.W[1], self.U[1], prev_state,  self.b[1]))
        h_t = tf.multiply(1 - u_t, prev_state) + tf.multiply(u_t, tf.tanh(
            r2n2_linear(fc_input, self.W[2], self.U[2], tf.multiply(r_t, prev_state), self.b[2])))

        return h_t


def r2n2_matmul(a, b):
    # print(a.shape, b.shape)
    ret = tf.expand_dims(a, axis=-2)
    # print(ret.shape, b.shape)
    ret = tf.matmul(ret, b)
    # print(ret.shape)
    ret = tf.squeeze(ret, axis=-2)
    # print(ret.shape)
    return ret


def r2n2_linear(x, W, U, h, b):
    # print(x.shape, W.shape, U.shape, h.shape, b.shape)
    Wx = tf.map_fn(lambda a: r2n2_matmul(a, W), x,
                   parallel_iterations=N_PARALLEL)
    Uh = tf.nn.conv3d(h, U, strides=[1, 1, 1, 1, 1], padding="SAME")
    # print(Wx.shape, Uh.shape, b.shape)
    return Wx + Uh + b


def r2n2_stack(x, N=4):
    return tf.transpose(tf.stack([tf.stack([tf.stack([x] * N)] * N)] * N), [3, 0, 1, 2, 4])
