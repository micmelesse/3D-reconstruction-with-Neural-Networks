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
from lib.utils import grep_epoch_name

import lib.vis as vis


# Recurrent Reconstruction Neural Network (R2N2)
class Network:
    def __init__(self, train_params=None):
        # read params
        if train_params is None:
            train_params = utils.read_params()['TRAIN_PARAMS']

        self.CREATE_TIME = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.MODEL_DIR = "out/model_{}".format(self.CREATE_TIME)
        utils.make_dir(self.MODEL_DIR)

        with open(self.MODEL_DIR + '/train_params.json', 'w') as f:
            json.dump({"TRAIN_PARAMS": train_params}, f)

        # place holders
        self.X = tf.placeholder(tf.float32, [None, 24, 137, 137, 4])
        X_drop_alpha = self.X[:, :, :, :, 0:3]
        X_cropped = tf.map_fn(lambda a: tf.random_crop(
            a, [24, 127, 127, 3]), X_drop_alpha)

        # encoder
        print("encoder")
        encoder = encoder_module.Conv_Encoder(X_cropped)
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
        self.Y = tf.placeholder(tf.uint8, [None, 32, 32, 32, 2])
        voxel_loss = loss_module.Voxel_Softmax(self.Y, self.logits)
        self.loss = voxel_loss.loss
        self.softmax = voxel_loss.softmax
        tf.summary.scalar("loss", self.loss)

        # optimizer
        print("optimizer")
        self.step_count = tf.Variable(
            0, trainable=False, name="step_count")

        if train_params["OPTIMIZER"] == "ADAM":
            optimizer = tf.train.AdamOptimizer()
            self.print = tf.Print(
                self.loss, [self.step_count, optimizer._lr, self.loss])
        else:
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=train_params["LEARN_RATE"])
            self.print = tf.Print(
                self.loss, [self.step_count, optimizer._learning_rate, self.loss])

        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.apply_grad = optimizer.apply_gradients(
            grads_and_vars, global_step=self.step_count)

        # misc op
        print("misc op")
        self.summary_op = tf.summary.merge_all()
        self.sess = tf.InteractiveSession()

        print("initalize variables")
        tf.global_variables_initializer().run()

        # pointers to training objects
        self.train_writer = tf.summary.FileWriter(
            "{}/train".format(self.MODEL_DIR), self.sess.graph)
        self.val_writer = tf.summary.FileWriter(
            "{}/val".format(self.MODEL_DIR), self.sess.graph)
        self.test_writer = tf.summary.FileWriter(
            "{}/test".format(self.MODEL_DIR), self.sess.graph)

    def step(self, data, label, step_type):
        utils.make_dir(self.MODEL_DIR)
        cur_dir = self.get_epoch_dir()
        x, y = dataset.from_npy(data), dataset.from_npy(label)

        if step_type == "train":
            out = self.sess.run([self.apply_grad, self.loss, self.summary_op,  self.print, self.step_count], {
                self.X: x, self.Y: y})
            self.train_writer.add_summary(out[2], global_step=out[4])
        else:
            out = self.sess.run([self.softmax, self.loss, self.summary_op, self.print, self.step_count], {
                self.X: x, self.Y: y})

            if step_type == "val":
                self.val_writer.add_summary(out[2], global_step=out[4])
            elif step_type == "test":
                self.test_writer.add_summary(out[2], global_step=out[4])

            i = np.random.randint(0, len(data))
            x_name = utils.get_file_name(data[i])
            y_name = utils.get_file_name(label[i])
            f_name = x_name[0:-2]
            sequence, voxel, softmax, step_count = x[i], y[i], out[0][i], out[4]

            # save plots
            vis.sequence(
                sequence, f_name="{}/{}_{}.png".format(cur_dir, step_count, x_name))
            vis.softmax(voxel,
                        f_name="{}/{}_{}.png".format(cur_dir, step_count, y_name))
            vis.softmax(
                softmax, f_name="{}/{}_{}_yp.png".format(cur_dir, step_count, f_name))
            np.save(
                "{}/{}_{}_sm.npy".format(cur_dir, step_count, f_name), softmax)

        return out[1]  # return the loss

    def save(self):
        cur_dir = self.get_epoch_dir()
        epoch_name = utils.grep_epoch_name(cur_dir)
        model_builder = tf.saved_model.builder.SavedModelBuilder(
            cur_dir + "/model")
        model_builder.add_meta_graph_and_variables(self.sess, [epoch_name])
        model_builder.save()

    def predict(self, x):
        return self.sess.run([self.softmax], {self.X: x})

    def get_params(self):
        utils.make_dir(self.MODEL_DIR)
        with open(self.MODEL_DIR+"/train_params.json") as fp:
            return json.load(fp)

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


class Network_restored:
    def __init__(self, model_dir):
        epoch_name = grep_epoch_name(model_dir)
        self.sess = tf.Session(graph=tf.Graph())
        tf.saved_model.loader.load(
            self.sess, [epoch_name], model_dir + "/model")

    def predict(self, x, in_name="Placeholder: 0", sm_name="loss/clip_by_value: 0"):
        if x.ndim == 4:
            x = np.expand_dims(x, 0)

        softmax = self.sess.graph.get_tensor_by_name(sm_name)
        in_tensor = self.sess.graph.get_tensor_by_name(in_name)
        return self.sess.run(softmax, {in_tensor: x})
