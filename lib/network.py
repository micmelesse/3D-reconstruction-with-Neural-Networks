import os
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import lib.utils as utils
import lib.dataset as dataset
import lib.encoder_module as encoder_module
import lib.recurrent_module as recurrent_module
import lib.decoder_module as decoder_module
import lib.loss_module as loss_module
import lib.optimizer_module as optimizer_module


# Recurrent Reconstruction Neural Network (R2N2)
class Network:
    def __init__(self, params=None):
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

        # create model_dir
        create_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        model_dir = "out/model_{}_L:{}_E:{}_B:{}".format(
            create_time, learn_rate, epoch_count, batch_size)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        # net variables
        self.model_dir = model_dir
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.epoch_count = epoch_count

        # place holders
        self.X = tf.placeholder(tf.float32, [None, 24, 137, 137, 4])

        # encoder
        print("encoder")
        encoder = encoder_module.Conv_Encoder(self.X)
        encoded_input = encoder.out_tensor

        print("recurrent_module")
        # recurrent_module
        with tf.name_scope("recurrent_module"):
            GRU_Grid = recurrent_module.GRU_Grid()
            hidden_state = None
            for t in range(24):
                hidden_state = GRU_Grid.call(
                    encoded_input[:, t, :], hidden_state)

        # decoder
        print("decoder")
        decoder = decoder_module.Conv_Decoder(hidden_state)
        logits = decoder.out_tensor

        # loss
        print("loss")
        self.Y = tf.placeholder(tf.uint8, [None, 32, 32, 32])
        voxel_softmax = loss_module.Voxel_Softmax(self.Y, logits)
        self.prediction = voxel_softmax.prediction
        self.loss = voxel_softmax.batch_loss
        tf.summary.scalar('loss', self.loss)

        # optimizer
        print("optimizer")
        sgd_optimizer = optimizer_module.SGD_optimizer(
            self.loss, learn_rate)
        self.apply_grad = sgd_optimizer.apply_grad

        # init network
        print("init session")
        self.print = tf.Print(
            self.loss, [sgd_optimizer.step_count, learn_rate, self.loss])
        self.summary_op = tf.summary.merge_all()
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

    def train_step(self, data, label):
        x = dataset.from_npy(data)
        y = dataset.from_npy(label)
        out = self.sess.run([self.loss, self.summary_op, self.apply_grad, self.print], {
            self.X: x, self.Y: y})
        writer = tf.summary.FileWriter(
            "{}/writer".format(self.model_dir), self.sess.graph)
        writer.add_summary(out[1])
        return out[0]

    def get_save_dir(self):
        i = 0
        while os.path.exists(os.path.join(self.model_dir, "epoch_{}".format(i))):
            i += 1
        save_dir = os.path.join(self.model_dir, "epoch_{}".format(i))
        os.makedirs(save_dir)
        return save_dir

    def restore(self):
        saver = tf.train.import_meta_graph(
            "{}/model.ckpt.meta".format(self.model_dir))
        saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))

    def predict(self, x):
        return self.sess.run([self.prediction], {self.X: x})[0]
