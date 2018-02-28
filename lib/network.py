import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import lib.utils as utils
import lib.encoder_module as encoder_module
import lib.recurrent_module as recurrent_module
import lib.decoder_module as decoder_module
from datetime import datetime

# Recurrent Reconstruction Neural Network (R2N2)


class R2N2:
    def __init__(self, params=None):
        self.session_loss = []
        self.create_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        # read params
        if params is None:
            self.learn_rate, self.batch_size, self.epoch_count = utils.get_params_from_disk()
            if self.learn_rate is None:
                self.learn_rate = 0.01
            if self.batch_size is None:
                self.batch_size = 16
            if self.epoch_count is None:
                self.epoch_count = 5

        else:
            self.learn_rate = params['learn_rate']
            self.batch_size = params['batch_size']
            self.epoch_count = params['epoch_count']

        print("learn_rate {}, epoch_count {}, batch_size {}".format(
            self.learn_rate, self.epoch_count, self.batch_size))

        # place holders
        print("creating network...")
        self.X = tf.placeholder(tf.float32, [None, 24, 137, 137, 4])

        print("encoder_network")
        with tf.name_scope("encoder_network"):
            encoder = encoder_module.Conv_Encoder(self.X)
            encoded_input = encoder.out_tensor

        print("recurrent_module")
        with tf.name_scope("recurrent_module"):
            GRU_Grid = recurrent_module.GRU_Grid()
            hidden_state = None
            for t in range(24):  # feed batches of seqeuences
                hidden_state = tf.verify_tensor_all_finite(GRU_Grid.call(
                    encoded_input[:, t, :], hidden_state), "hidden_state {}".format(t))

        print("decoder_network")
        with tf.name_scope("decoder_network"):
            decoder = decoder_module.Conv_Decoder(hidden_state)
            logits = decoder.out_tensor

        self.Y = tf.placeholder(tf.uint8, [None, 32, 32, 32])
        print("loss_function")
        with tf.name_scope("loss"):
            softmax = tf.nn.softmax(logits)
            log_softmax = tf.nn.log_softmax(logits)  # avoids log(0)
            label = tf.one_hot(self.Y, 2)
            cross_entropy = tf.reduce_sum(-tf.multiply(label,
                                                       log_softmax), axis=-1)
            losses = tf.reduce_mean(cross_entropy, axis=[1, 2, 3])
            batch_loss = tf.reduce_mean(losses)
            tf.summary.scalar("loss", batch_loss)
            self.loss = batch_loss

        print("update functions")
        with tf.name_scope("update"):
            step_count = tf.Variable(0, trainable=False)
            lr = self.learn_rate
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=lr)
            grads_and_vars = optimizer.compute_gradients(batch_loss)
            map(lambda a: tf.verify_tensor_all_finite(
                a[0], "grads_and_vars"), grads_and_vars)  # assert no Nan or Infs in grad

            self.apply_grad = optimizer.apply_gradients(
                grads_and_vars, global_step=step_count)
            self.summary_op = tf.summary.merge_all()
            self.print = tf.Print(batch_loss, [step_count, batch_loss, lr])

        print("...network created")
        with tf.name_scope("misc"):
            self.saver = tf.train.Saver()
            self.sess = tf.InteractiveSession()
            self.prediction = tf.argmax(softmax, -1)
            tf.global_variables_initializer().run()

    def train_step(self, data, label):
        x = utils.to_npy(data)
        y = utils.to_npy(label)
        return self.sess.run([self.apply_grad, self.summary_op, self.print, self.loss], {self.X: x, self.Y: y})[-1]

    def save(self, save_dir):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.saver.save(self.sess, "{}/model.ckpt".format(save_dir))
        np.save("{}/loss.npy".format(save_dir), self.session_loss)
        self.plot_loss(save_dir, self.session_loss)
        writer = tf.summary.FileWriter(
            "{}/writer".format(save_dir), self.sess.graph)

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
