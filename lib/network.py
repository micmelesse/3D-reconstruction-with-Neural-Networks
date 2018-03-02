import os
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import lib.utils as utils
import lib.encoder_module as encoder_module
import lib.recurrent_module as recurrent_module
import lib.decoder_module as decoder_module
import lib.loss_module as loss_module
import lib.optimizer_module as optimizer_module


# Recurrent Reconstruction Neural Network (R2N2)
class reconstruction_network:
    def __init__(self, params=None):
        self.session_loss = []
        self.create_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        # read params
        if params is None:
            learn_rate, batch_size, epoch_count = utils.get_params_from_disk()
            if learn_rate is None:
                learn_rate = 0.01
            if batch_size is None:
                batch_size = 16
            if epoch_count is None:
                epoch_count = 5
        else:
            learn_rate = params['learn_rate']
            batch_size = params['batch_size']
            epoch_count = params['epoch_count']

        print("learn_rate {}, epoch_count {}, batch_size {}".format(
            learn_rate, epoch_count, batch_size))

        # place holders
        self.X = tf.placeholder(tf.float32, [None, 24, 137, 137, 4])

        # encoder
        encoder = encoder_module.Conv_Encoder(self.X)
        encoded_input = encoder.out_tensor

        # recurrent_module
        GRU_Grid = recurrent_module.GRU_Grid()
        hidden_state = None
        for t in range(24):
            hidden_state = GRU_Grid.call(
                encoded_input[:, t, :], hidden_state)

        # decoder
        decoder = decoder_module.Conv_Decoder(hidden_state)
        logits = decoder.out_tensor

        # loss
        self.Y = tf.placeholder(tf.uint8, [None, 32, 32, 32])
        voxel_softmax = loss_module.Voxel_Softmax(self.Y, logits)
        self.prediction = voxel_softmax.prediction
        self.loss = voxel_softmax.batch_loss

        # optimizer
        sgd_optimizer = optimizer_module.SGD_optimizer(
            self.loss, learn_rate)
        self.apply_grad = sgd_optimizer.apply_grad

        self.print = tf.Print(
            self.loss, [sgd_optimizer.step_count, learn_rate, self.loss])
        self.summary_op = tf.summary.merge_all()

        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

    def train_step(self, data, label):
        x = utils.to_npy(data)
        y = utils.to_npy(label)
        return self.sess.run([self.loss, self.apply_grad, self.summary_op, self.print], {self.X: x, self.Y: y})[0]

    def save(self, save_dir):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        self.plot_loss(save_dir, self.session_loss)
        np.save("{}/loss.npy".format(save_dir), self.session_loss)
        writer = tf.summary.FileWriter(
            "{}/writer".format(save_dir), self.sess.graph)
        tf.train.Saver().save(self.sess, "{}/model.ckpt".format(save_dir))

    def restore(self, model_dir):
        saver = tf.train.import_meta_graph(
            "{}/model.ckpt.meta".format(model_dir))
        saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))

    def predict(self, x):
        return self.sess.run([self.prediction], {self.X: x})[0]

    def plot_loss(self, plot_dir, loss_arr):
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)
        plt.plot(np.array(loss_arr).flatten())
        plt.savefig("{}/loss.png".format(plot_dir), bbox_inches='tight')
        plt.close()
