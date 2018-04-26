import os
import sys
import re
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from lib import dataset, preprocessor, encoder, recurrent_module, decoder, loss, vis, utils


# Recurrent Reconstruction Neural Network (R2N2)
class Network:
    def __init__(self, params=None):
        # read params
        if params is None:
            self.params = utils.read_params()
        else:
            self.params = params

        if self.params["TRAIN_PARAMS"]["INITIALIZER"] == "XAVIER":
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = tf.random_normal_initializer()

        self.CREATE_TIME = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.MODEL_DIR = "{}/model_{}".format(
            self.params["DIRS"]["MODELS_LOCAL"], self.CREATE_TIME)
        utils.make_dir(self.MODEL_DIR)

        with open(self.MODEL_DIR + '/params.json', 'w') as f:
            json.dump(self.params, f)

        # place holders
        self.X = tf.placeholder(tf.float32, [None, None, 137, 137, 4])
        pp = preprocessor.Preprocessor(self.X)
        X_preprocessed = pp.out_tensor

        # encoder
        print("encoder")
        en = encoder.Basic_Encoder(X_preprocessed)
        encoded_input = en.out_tensor

        print("recurrent_module")
        # recurrent_module
        with tf.name_scope("recurrent_module"):
            n_batchsize = tf.shape(self.X)[0]
            hidden_state = tf.zeros([n_batchsize, 4, 4, 4, 128])
            GRU_Grid = recurrent_module.GRU_Grid(initializer=init)
            for t in range(24):
                hidden_state = GRU_Grid.call(
                    encoded_input[:, t, :], hidden_state)

            # GRU_Grid = recurrent_module.GRU_Grid(initializer=init)
            # hidden_state = tf.zeros(
            #     [n_batchsize, 4, 4, 4, 128])

            # i = tf.constant(0)`

            # def condition(h, i):
            #     return tf.less(i, n_timesteps)

            # def body(h, i):
            #
            #     hidden_state = GRU_Grid.call(
            #         encoded_input[:, i, :], h)
            #       tf.add(i, 1)
            #     return hidden_state, i

            # hidden_state, i = tf.while_loop(condition, body, [hidden_state, i])

        # decoder
        print("decoder")
        de = decoder.Basic_Decoder_old(hidden_state)
        self.logits = de.out_tensor

        # loss
        print("loss")
        self.Y_onehot = tf.placeholder(tf.float32, [None, 32, 32, 32, 2])
        voxel_loss = loss.Voxel_Softmax(self.Y_onehot, self.logits)
        self.loss = voxel_loss.loss
        self.softmax = voxel_loss.softmax
        tf.summary.scalar("loss", self.loss)

        # optimizer
        print("optimizer")
        self.step_count = tf.Variable(
            0, trainable=False, name="step_count")
        if self.params["TRAIN_PARAMS"]["OPTIMIZER"] == "ADAM":
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.params["TRAIN_PARAMS"]["ADAM_LEARN_RATE"], epsilon=self.params["TRAIN_PARAMS"]["ADAM_EPSILON"])
            tf.summary.scalar("adam_learning_rate", optimizer._lr)
        else:
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self.params["TRAIN_PARAMS"]["LEARN_RATE"])
            tf.summary.scalar("learning_rate", optimizer._learning_rate)

        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.apply_grad = optimizer.apply_gradients(
            grads_and_vars, global_step=self.step_count)

        # trainable_parameters = tf.trainable_variables()
        # for p in trainable_parameters:
        #     tf.summary.histogram(p.name, p)

        # metric
        print("metrics")
        with tf.name_scope("metrics"):
            Y = tf.argmax(self.Y_onehot, -1)
            predictions = tf.argmax(self.softmax, -1)
            acc, self.acc_op = tf.metrics.accuracy(Y, predictions)
            rms, self.rms_op = tf.metrics.root_mean_squared_error(
                self.Y_onehot, self.softmax)
            iou, self.iou_op = tf.metrics.mean_iou(Y, predictions, 2)

        tf.summary.scalar("accuracy", acc)
        tf.summary.scalar("rmse", rms)
        tf.summary.scalar("iou", iou)

        # initalize
        print("initalize")
        self.print = tf.Print(self.loss, [self.step_count, self.loss])
        self.summary_op = tf.summary.merge_all()
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        # summaries
        print("summaries")
        if self.params["MODE"] == "TEST":
            self.test_writer = tf.summary.FileWriter(
                "{}/test".format(self.MODEL_DIR), self.sess.graph)
        else:
            self.train_writer = tf.summary.FileWriter(
                "{}/train".format(self.MODEL_DIR), self.sess.graph)
            self.val_writer = tf.summary.FileWriter(
                "{}/val".format(self.MODEL_DIR), self.sess.graph)

        print("...done!")

    def step(self, data, label, step_type):
        utils.make_dir(self.MODEL_DIR)
        cur_dir = self.get_cur_epoch_dir()
        data_npy, label_npy = utils.load_npy(data), utils.load_npy(label)

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
            if self.params["TRAIN_PARAMS"]["VIS_VALIDATION"]:
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
        with open(self.MODEL_DIR+"/params.json") as fp:
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

    def feature_maps(self, x):
        pass
