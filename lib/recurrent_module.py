import os
import sys
import math
import numpy as np
import tensorflow as tf
from lib import utils


class GRU_Grid:
    def __init__(self,  n_cells=4,  n_input=1024, n_hidden_state=128, initializer=None):
        with tf.name_scope("GRU_Grid"):
            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            # parameters for the hidden state and update & reset gates
            self.W = [Weight_Matrices(
                n_cells, n_input, n_hidden_state, initializer=init)]*3
            self.U = [tf.Variable(init(
                [3, 3, 3, n_hidden_state, n_hidden_state]), name="U")]*3
            self.b = [tf.Variable(init(
                [n_cells, n_cells, n_cells, n_hidden_state]), name="b")]*3

            params = utils.read_params()
            if params["VIS"]["HISTOGRAMS"]:
                for i in range(3):
                    tf.summary.histogram("U[{}]".format(i), self.U[i])
                    tf.summary.histogram("b[{}]".format(i), self.b[i])

    def pre_activity(self, W, x, U, h, b):
        return W.multiply_grid(x) + tf.nn.conv3d(h, U, strides=[1, 1, 1, 1, 1], padding="SAME") + b

    def call(self, input_tensor, prev_hidden):
        # update gate
        u_t = tf.sigmoid(
            self.pre_activity(self.W[0], input_tensor, self.U[0], prev_hidden, self.b[0]))
        # reset gate
        r_t = tf.sigmoid(
            self.pre_activity(self.W[1], input_tensor, self.U[1], prev_hidden,  self.b[1]))

        # hidden state
        h_t_1 = (1 - u_t) * prev_hidden
        h_t_2 = u_t * tf.tanh(self.pre_activity(self.W[2], input_tensor,
                                                self.U[2], r_t * prev_hidden, self.b[2]))
        h_t = h_t_1 + h_t_2
        return h_t


class LSTM_Grid:
    def __init__(self, n_cells=4, n_input=1024, n_hidden_state=128, initializer=None):
        with tf.name_scope("LSTM_Grid"):
            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            # parameters for the cell state and input,forget & output gates
            self.W = [Weight_Matrices(
                n_cells, n_input, n_hidden_state, initializer=init)]*4
            self.U = [tf.Variable(init(
                [3, 3, 3, n_hidden_state, n_hidden_state]), name="U")]*4
            self.b = [tf.Variable(init(
                [n_cells, n_cells, n_cells, n_hidden_state]), name="b")]*4

            params = utils.read_params()
            if params["VIS"]["HISTOGRAMS"]:
                for i in range(4):
                    tf.summary.histogram("U[{}]".format(i), self.U[i])
                    tf.summary.histogram("b[{}]".format(i), self.b[i])

    def pre_activity(self, W, x, U, h, b):
        return W.multiply_grid(x) + tf.nn.conv3d(h, U, strides=[1, 1, 1, 1, 1], padding="SAME") + b

    def call(self, input_tensor, prev_state):
        prev_hidden_state, prev_cell_state = prev_state

        # forget gate
        f_t = tf.sigmoid(
            self.pre_activity(self.W[0], input_tensor, self.U[0], prev_hidden_state, self.b[0]))

        # input gate
        i_t = tf.sigmoid(
            self.pre_activity(self.W[1], input_tensor, self.U[1], prev_hidden_state,  self.b[1]))

        # output gate
        o_t = tf.sigmoid(
            self.pre_activity(self.W[2], input_tensor, self.U[2], prev_hidden_state, self.b[2]))

        # cell state
        s_t_1 = f_t * prev_cell_state
        s_t_2 = i_t * tf.tanh(self.pre_activity(self.W[3], input_tensor,
                                                self.U[3], prev_hidden_state, self.b[3]))
        s_t = s_t_1 + s_t_2
        h_t = o_t*tf.tanh(s_t)

        return (h_t, s_t)


class Weight_Matrices:
    def __init__(self,  n_cells, n_x, n_h,  initializer=None):
        with tf.name_scope("Weight_Matrices"):
            params = utils.read_params()
            # class variables
            self.n_cells = n_cells

            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            with tf.name_scope("x_list"):
                x_list = []
                for x in range(self.n_cells):
                    with tf.name_scope("y_list"):
                        y_list = []
                        for y in range(self.n_cells):
                            z_list = []
                            with tf.name_scope("z_list"):
                                for z in range(self.n_cells):
                                    name = "W_{}{}{}".format(x, y, z)
                                    W = tf.Variable(init(
                                        [n_x, n_h]), name=name)

                                    if params["VIS"]["HISTOGRAMS"]:
                                        tf.summary.histogram(name, W)
                                    z_list.append(W)
                            y_list.append(z_list)
                    x_list.append(y_list)

            self.weight_matrix_grid = x_list

    # multiply each of weight matrix with x
    def multiply_grid(self, x):
        with tf.name_scope("multiply_grid"):
            x_list = []
            for i in range(self.n_cells):
                y_list = []
                for j in range(self.n_cells):
                    z_list = []
                    for k in range(self.n_cells):
                        transformed_vector = tf.matmul(
                            x, self.weight_matrix_grid[i][j][k])
                        z_list.append(transformed_vector)
                    y_list.append(z_list)
                x_list.append(y_list)

        return tf.transpose(tf.convert_to_tensor(x_list), [3, 0, 1, 2, 4])
