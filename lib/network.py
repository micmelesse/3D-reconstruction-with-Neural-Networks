import os
import sys
import re
import json
import math
import random
import keyboard
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from lib import dataset, preprocessor, encoder, recurrent_module, decoder, loss, vis, utils
from tensorflow.python import debug as tf_debug


# Recurrent Reconstruction Neural Network (R2N2)
class Network:
    def __init__(self, params=None):
        # read params
        if params is None:
            self.params = utils.read_params()
        else:
            self.params = params

        if self.params["TRAIN"]["INITIALIZER"] == "XAVIER":
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
        with tf.name_scope("Data"):
            self.X = tf.placeholder(tf.float32, [None, None, 137, 137, 4])
        with tf.name_scope("Labels"):
            self.Y_onehot = tf.placeholder(tf.float32, [None, 32, 32, 32, 2])

        pp = preprocessor.Preprocessor(self.X)
        X_preprocessed = pp.out_tensor
        n_batchsize = tf.shape(X_preprocessed)[0]

        # encoder
        print("encoder")
        if self.params["TRAIN"]["ENCODER_MODE"] == "DILATED":
            en = encoder.Dilated_Encoder(X_preprocessed)
        elif self.params["TRAIN"]["ENCODER_MODE"] == "RESIDUAL":
            en = encoder.Residual_Encoder(X_preprocessed)
        else:
            en = encoder.Simple_Encoder(X_preprocessed)
        encoded_input = en.out_tensor

        print("recurrent_module")
        # recurrent_module
        with tf.name_scope("recurrent_module"):
            if self.params["TRAIN"]["RNN_MODE"] == "LSTM":
                rnn = recurrent_module.LSTM_Grid(initializer=init)
                hidden_state = (tf.zeros([n_batchsize, 4, 4, 4, 128], name="zero_hidden_state"), tf.zeros(
                    [n_batchsize, 4, 4, 4, 128], name="zero_cell_state"))
            else:
                rnn = recurrent_module.GRU_Grid(initializer=init)
                hidden_state = tf.zeros(
                    [n_batchsize, 4, 4, 4, 128], name="zero_hidden_state")

            # feed a limited seqeuence of images
            if self.params["TRAIN"]["TIME_STEPS"] > 0:
                n_timesteps = self.params["TRAIN"]["TIME_STEPS"]
                for t in range(n_timesteps):
                    hidden_state = rnn.call(
                        encoded_input[:, t, :], hidden_state)
            else:
                n_timesteps = tf.shape(X_preprocessed)[1]

                t = tf.constant(0)

                def condition(h, t):
                    return tf.less(t, n_timesteps)

                def body(h, t):
                    h = rnn.call(
                        encoded_input[:, t, :], h)
                    tf.add(t, 1)
                    return h, t

                hidden_state, t = tf.while_loop(
                    condition, body, (hidden_state, t), maximum_iterations=25)

        # decoder
        print("decoder")
        if isinstance(hidden_state, tuple):
            hidden_state = hidden_state[0]

        if self.params["TRAIN"]["DECODER_MODE"] == "DILATED":
            de = decoder.Dilated_Decoder(hidden_state)
        elif self.params["TRAIN"]["DECODER_MODE"] == "RESIDUAL":
            de = decoder.Residual_Decoder(hidden_state)
        else:
            de = decoder.Simple_Decoder(hidden_state)

        self.logits = de.out_tensor

        # loss
        print("loss")
        voxel_loss = loss.Voxel_Softmax(self.Y_onehot, self.logits)
        self.loss = voxel_loss.loss
        self.softmax = voxel_loss.softmax
        tf.summary.scalar("loss", self.loss)

        # misc
        print("misc")
        with tf.name_scope("misc"):
            self.step_count = tf.Variable(
                0, trainable=False, name="step_count")
            self.print = tf.Print(self.loss, [self.step_count, self.loss])

        # optimizer
        print("optimizer")
        if self.params["TRAIN"]["OPTIMIZER"] == "ADAM":
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.params["TRAIN"]["ADAM_LEARN_RATE"], epsilon=self.params["TRAIN"]["ADAM_EPSILON"])
            tf.summary.scalar("adam_learning_rate", optimizer._lr)
        else:
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self.params["TRAIN"]["GD_LEARN_RATE"])
            tf.summary.scalar("learning_rate", optimizer._learning_rate)

        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.apply_grad = optimizer.apply_gradients(
            grads_and_vars, global_step=self.step_count)

        # metric
        print("metrics")
        with tf.name_scope("metrics"):
            Y = tf.argmax(self.Y_onehot, -1)
            predictions = tf.argmax(self.softmax, -1)
            acc, acc_op = tf.metrics.accuracy(Y, predictions)
            rms, rms_op = tf.metrics.root_mean_squared_error(
                self.Y_onehot, self.softmax)
            iou, iou_op = tf.metrics.mean_iou(Y, predictions, 2)
            self.metrics_op = tf.group(acc_op, rms_op, iou_op)

        tf.summary.scalar("accuracy", acc)
        tf.summary.scalar("rmse", rms)
        tf.summary.scalar("iou", iou)

        # initalize
        # config=tf.ConfigProto(log_device_placement=True)
        print("setup")
        self.summary_op = tf.summary.merge_all()
        self.sess = tf.InteractiveSession()
        if self.params["MODE"] == "DEBUG":
            self.sess = tf_debug.TensorBoardDebugWrapperSession(
                self.sess, "nat-oitwireless-inside-vapornet100-c-15126.Princeton.EDU:6064")

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

        # initialize
        print("initialize")
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        print("...done!")

    def step(self, data, label, step_type):
        utils.make_dir(self.MODEL_DIR)
        cur_dir = self.get_cur_epoch_dir()
        data_npy, label_npy = utils.load_npy(data), utils.load_npy(label)
        feed_dict = {self.X: data_npy, self.Y_onehot: label_npy}

        if step_type == "train":
            fetches = [self.apply_grad, self.loss, self.summary_op,
                       self.print, self.step_count, self.metrics_op]
            out = self.sess.run(fetches, feed_dict)
            loss, summary, step_count = out[1], out[2], out[4]

            self.train_writer.add_summary(summary, global_step=step_count)
        elif step_type == "debug":
            fetchs = [self.apply_grad]
            options = tf.RunOptions(trace_level=3)
            run_metadata = tf.RunMetadata()
            out = self.sess.run(fetches, feed_dict,
                                options=options, run_metadata=run_metadata)
        else:
            fetchs = [self.softmax, self.loss, self.summary_op, self.print,
                      self.step_count, self.metrics_op]
            out = self.sess.run(fetchs, feed_dict)
            softmax, loss, summary, step_count = out[0], out[1], out[2], out[4]

            if step_type == "val":
                self.val_writer.add_summary(summary, global_step=step_count)
            elif step_type == "test":
                self.test_writer.add_summary(summary, global_step=step_count)

            # display the result of each element of the validation batch
            if self.params["VIS"]["VALIDATION_STEP"]:
                i = random.randint(0, len(data_npy)-1)
                x, y, yp = data_npy[i], label_npy[i], softmax[i]
                name = "{}/{}_{}".format(cur_dir, step_count,
                                         utils.get_file_name(data[i])[0:-2])
                vis.voxel_binary(yp, "{}.png".format(name))

        return loss

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
