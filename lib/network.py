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
        self.step_count = tf.Variable(
            0, trainable=False, name="step_count")
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learn_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.apply_grad = optimizer.apply_gradients(
            grads_and_vars, global_step=self.step_count)

        # misc op
        print("misc op")
        self.print = tf.Print(
            self.loss, [self.step_count, learn_rate, self.loss])
        self.summary_op = tf.summary.merge_all()
        self.sess = tf.InteractiveSession()
        self.train_writer = tf.summary.FileWriter(
            "{}/train".format(self.model_dir), self.sess.graph)
        self.val_writer = tf.summary.FileWriter(
            "{}/val".format(self.model_dir), self.sess.graph)
        self.test_writer = tf.summary.FileWriter(
            "{}/test".format(self.model_dir), self.sess.graph)

        # init network
        print("initalize variables")
        tf.global_variables_initializer().run()

    def step(self, data, label, step_type):
        x = dataset.from_npy(data)
        y = dataset.from_npy(label)

        if step_type == "train":
            out = self.sess.run([self.loss, self.summary_op, self.apply_grad, self.print, self.step_count], {
                self.X: x, self.Y: y})
            self.train_writer.add_summary(out[1], out[4])
        else:
            out = self.sess.run([self.loss, self.summary_op, self.print, self.step_count], {
                self.X: x, self.Y: y})
            if step_type == "val":
                self.val_writer.add_summary(out[1], out[3])
            elif step_type == "test":
                self.test_writer.add_summary(out[1], out[3])

        # return the loss
        return out[0]

    def create_epoch_dir(self):
        new_ind = self.epoch_index()+1
        save_dir = os.path.join(self.model_dir, "epoch_{}".format(new_ind))
        os.makedirs(save_dir)
        return save_dir

    def get_epoch_dir(self):
        cur_ind = self.epoch_index()
        save_dir = os.path.join(
            self.model_dir, "epoch_{}".format(cur_ind))
        return save_dir

    def epoch_index(self):
        i = 0
        while os.path.exists(os.path.join(self.model_dir, "epoch_{}".format(i))):
            i += 1
        return i-1

    def restore(self):
        saver = tf.train.import_meta_graph(
            "{}/model.ckpt.meta".format(self.model_dir))
        saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))

    def predict(self, x):
        return self.sess.run([self.prediction], {self.X: x})[0]
