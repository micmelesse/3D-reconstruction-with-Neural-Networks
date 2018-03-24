import os
import sys
import re
import json
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
            params = utils.read_params()['TRAIN_PARAMS']

        self.LEARN_RATE = params['LEARN_RATE']
        self.BATCH_SIZE = params['BATCH_SIZE']
        self.EPOCH_COUNT = params['EPOCH_COUNT']
        self.CREATE_TIME = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.MODEL_DIR = "out/model_{}_L:{}_E:{}_B:{}".format(
            self.CREATE_TIME, self.LEARN_RATE, self.EPOCH_COUNT, self.BATCH_SIZE)

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
        self.logits = decoder.out_tensor

        # loss
        print("loss")
        self.Y = tf.placeholder(tf.uint8, [None, 32, 32, 32])
        with tf.name_scope("loss"):
            self.softmax = tf.nn.softmax(self.logits)
            log_softmax = tf.nn.log_softmax(self.logits)  # avoids log(0)
            label = tf.one_hot(self.Y, 2)
            cross_entropy = tf.reduce_sum(-tf.multiply(label,
                                                       log_softmax), axis=-1)
            losses = tf.reduce_mean(cross_entropy, axis=[1, 2, 3])
            self.prediction = tf.argmax(self.softmax, -1)
            self.loss = tf.reduce_mean(losses)

        tf.summary.scalar('loss', self.loss)

        # optimizer
        print("optimizer")
        self.step_count = tf.Variable(
            0, trainable=False, name="step_count")
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.LEARN_RATE)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.apply_grad = optimizer.apply_gradients(
            grads_and_vars, global_step=self.step_count)

        # misc op
        print("misc op")
        self.print = tf.Print(
            self.loss, [self.step_count, self.LEARN_RATE, self.loss])
        self.summary_op = tf.summary.merge_all()
        self.sess = tf.InteractiveSession()

        print("initalize variables")
        tf.global_variables_initializer().run()

        # pointers to training objects
        self.train_writer = None
        self.val_writer = None
        self.test_writer = None

    # init network
    def init(self):
        utils.make_dir(self.MODEL_DIR)
        self.train_writer = tf.summary.FileWriter(
            "{}/train".format(self.MODEL_DIR), self.sess.graph)
        self.val_writer = tf.summary.FileWriter(
            "{}/val".format(self.MODEL_DIR), self.sess.graph)
        self.test_writer = tf.summary.FileWriter(
            "{}/test".format(self.MODEL_DIR), self.sess.graph)

    def step(self, data, label, step_type):
        x = dataset.from_npy(data)
        y = dataset.from_npy(label)
        cur_dir = self.get_epoch_dir()
        if step_type == "train":
            out = self.sess.run([self.loss, self.summary_op, self.apply_grad, self.print, self.step_count], {
                self.X: x, self.Y: y})
            self.train_writer.add_summary(out[1], out[4])
        else:
            out = self.sess.run([self.loss, self.summary_op, self.print, self.step_count, self.prediction], {
                self.X: x, self.Y: y})

            utils.vis_validation(x, y, out[4], cur_dir)
            if step_type == "val":
                self.val_writer.add_summary(out[1], out[3])
            elif step_type == "test":
                self.test_writer.add_summary(out[1], out[3])

        # return the loss
        return out[0]

    def save(self):
        cur_dir = self.get_epoch_dir()
        epoch_name = utils.grep_epoch_name(cur_dir)
        model_builder = tf.saved_model.builder.SavedModelBuilder(
            cur_dir + "/model")
        model_builder.add_meta_graph_and_variables(self.sess, [epoch_name])
        model_builder.save()

    def restore(self, epoch_dir):
        epoch_name = utils.grep_epoch_name(epoch_dir)
        tf.saved_model.loader.load(
            self.sess, [epoch_name], epoch_dir + "/model")

    def predict(self, x):
        return self.sess.run([self.prediction, self.softmax], {self.X: x})

    def info(self):
        print("LEARN_RATE:{}".format(
            self.LEARN_RATE))
        print("EPOCH_COUNT:{}".format(
            self.EPOCH_COUNT))
        print("BATCH_SIZE:{}".format(
            self.BATCH_SIZE))

    def create_epoch_dir(self):
        cur_ind = self.epoch_index()
        save_dir = os.path.join(self.MODEL_DIR, "epoch_{}".format(cur_ind+1))
        utils.make_dir(save_dir)
        return save_dir

    def get_epoch_dir(self):
        cur_ind = self.epoch_index()
        save_dir = os.path.join(
            self.MODEL_DIR, "epoch_{}".format(cur_ind))
        return save_dir

    def epoch_index(self):
        i = 0
        while os.path.exists(os.path.join(self.MODEL_DIR, "epoch_{}".format(i))):
            i += 1
        return i-1
