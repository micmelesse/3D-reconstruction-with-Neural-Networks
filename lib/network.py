import os
import sys
import re
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from lib import dataset, encoder, recurrent_module, decoder, loss, vis, utils


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
        self.X = tf.placeholder(tf.float32, [None, None, 137, 137, 4])
        n_timesteps = tf.shape(self.X)[1]
        X_drop_alpha = self.X[:, :, :, :, 0:3]
        X_cropped = tf.map_fn(lambda a: tf.random_crop(
            a, [n_timesteps, 127, 127, 3]), X_drop_alpha)

        # encoder
        print("encoder")
        en = encoder.Original_Encoder(X_cropped)
        encoded_input = en.out_tensor

        print("recurrent_module")
        # recurrent_module
        with tf.name_scope("recurrent_module"):
            GRU_Grid = recurrent_module.GRU_Grid()
            hidden_state = None
            for t in range(24):
                hidden_state = GRU_Grid.call(
                    encoded_input[:, t, :], hidden_state)
            
            # i = tf.constant(0)
            # def condition(i):
            #     return tf.less(i, n_timesteps)

            # def body(i):
            #     tf.add(i, 1)
            #     return GRU_Grid.call(
            #         encoded_input[:, i, :], hidden_state)
            # hidden_state = tf.while_loop(condition, body, [i])

        # decoder
        print("decoder")
        de = decoder.Original_Decoder(hidden_state)
        self.logits = de.out_tensor

        # loss
        print("loss")
        self.Y_onehot = tf.placeholder(tf.float32, [None, 32, 32, 32, 2])
        voxel_loss = loss.Voxel_Softmax(self.Y_onehot, self.logits)
        self.loss = voxel_loss.loss
        self.softmax = voxel_loss.softmax
        tf.summary.scalar("loss", self.loss)

        # metric
        print("metrics")
        Y = tf.argmax(self.Y_onehot, -1)
        predictions = tf.argmax(self.softmax, -1)
        acc, self.acc_op = tf.metrics.accuracy(Y, predictions)
        rms, self.rms_op = tf.metrics.root_mean_squared_error(
            self.Y_onehot, self.softmax)
        iou, self.iou_op = tf.metrics.mean_iou(Y, predictions, 2)

        tf.summary.scalar("accuracy", acc)
        tf.summary.scalar("rmse", rms)
        tf.summary.scalar("iou", iou)

        # optimizer
        print("optimizer")
        self.step_count = tf.Variable(
            0, trainable=False, name="step_count")
        if train_params["OPTIMIZER"] == "ADAM":
            optimizer = tf.train.AdamOptimizer(
                learning_rate=train_params["LEARN_RATE"], epsilon=train_params["ADAM_EPSILON"])
        else:
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=train_params["LEARN_RATE"])

        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.apply_grad = optimizer.apply_gradients(
            grads_and_vars, global_step=self.step_count)

        params = tf.trainable_variables()
        for p in params:
            tf.summary.histogram(p.name, p)

        # misc op
        print("misc op")
        self.print = tf.Print(self.loss, [self.step_count, self.loss])
        self.summary_op = tf.summary.merge_all()
        self.sess = tf.InteractiveSession()

        # pointers to summary objects
        print("summary writers")
        self.train_writer = tf.summary.FileWriter(
            "{}/train".format(self.MODEL_DIR), self.sess.graph)
        self.val_writer = tf.summary.FileWriter(
            "{}/val".format(self.MODEL_DIR), self.sess.graph)
        self.test_writer = tf.summary.FileWriter(
            "{}/test".format(self.MODEL_DIR), self.sess.graph)

        print("initalize variables")
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

    def step(self, data, label, step_type, vis_validation=False):
        utils.make_dir(self.MODEL_DIR)
        cur_dir = self.get_cur_epoch_dir()
        data_npy, label_npy = dataset.from_npy(data), dataset.from_npy(label)

        if step_type == "train":
            out = self.sess.run([self.apply_grad, self.loss, self.summary_op,  self.print, self.step_count,
                                 self.acc_op, self.iou_op, self.rms_op], {self.X: data_npy, self.Y_onehot: label_npy})
            self.train_writer.add_summary(out[2], global_step=out[4])
        else:
            out = self.sess.run([self.softmax, self.loss, self.summary_op, self.print, self.step_count,
                                 self.acc_op, self.iou_op, self.rms_op], {self.X: data_npy, self.Y_onehot: label_npy})

            if step_type == "val":
                self.val_writer.add_summary(out[2], global_step=out[4])
            elif step_type == "test":
                self.test_writer.add_summary(out[2], global_step=out[4])

            step_count = out[4]
            # display the result of each element of the validation batch
            if vis_validation:
                for x, y, yp, name in zip(data_npy, label_npy, out[0], data):
                    f_name = utils.get_file_name(name)[0:-2]
                    vis.img_sequence(
                        x, f_name="{}/{}_{}_x.png".format(cur_dir, step_count, f_name))
                    vis.voxel_binary(
                        y, f_name="{}/{}_{}_y.png".format(cur_dir, step_count, f_name))
                    vis.voxel_binary(
                        yp, f_name="{}/{}_{}_yp.png".format(cur_dir, step_count, f_name))
                    np.save(
                        "{}/{}_{}_yp.npy".format(cur_dir, step_count, f_name), yp)

        return out[1]  # return the loss

    def save(self):
        cur_dir = self.get_cur_epoch_dir()
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

    def get_cur_epoch_dir(self):
        cur_ind = self.epoch_index()
        save_dir = os.path.join(
            self.MODEL_DIR, "epoch_{}".format(cur_ind))
        return save_dir

    def epoch_index(self):
        model_info = utils.get_model_info(self.MODEL_DIR)
        return model_info["EPOCH_INDEX"]


class Network_restored:
    def __init__(self, model_dir):
        epoch_name = utils.grep_epoch_name(model_dir)
        self.sess = tf.Session(graph=tf.Graph())
        tf.saved_model.loader.load(
            self.sess, [epoch_name], model_dir + "/model")

    def predict(self, x, in_name="Placeholder: 0", sm_name="loss/clip_by_value: 0"):
        if x.ndim == 4:
            x = np.expand_dims(x, 0)

        softmax = self.sess.graph.get_tensor_by_name(sm_name)
        in_tensor = self.sess.graph.get_tensor_by_name(in_name)
        return self.sess.run(softmax, {in_tensor: x})
