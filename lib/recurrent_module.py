import os
import sys
import math
import numpy as np
import tensorflow as tf
from lib import utils


class GRU_Grid:
    def __init__(self, initializer=None, N=3, n_cells=4, n_input=1024, n_hidden_state=128):
        with tf.name_scope("GRU_Grid"):
            self.N = 3
            self.n_cells = 4
            self.n_input = 1024
            self.n_hidden_state = 128

            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            self.W = [Weight_Matrix_Grid(initializer=init)]*N
            self.U = [tf.Variable(init(
                [3, 3, 3, n_hidden_state, n_hidden_state]), name="U")]*N
            self.b = [tf.Variable(init(
                [n_cells, n_cells, n_cells, n_hidden_state]), name="b")]*N

            for i in range(N):
                tf.summary.histogram("U[{}]".format(i), self.U[i])
                tf.summary.histogram("b[{}]".format(i), self.b[i])

    def linear_sum(self, W, x, U, h, b):
        return W.multiply_grid(x) + tf.nn.conv3d(h, U, strides=[1, 1, 1, 1, 1], padding="SAME") + b

    def call(self, fc_input, prev_hidden):
        # update gate
        u_t = tf.sigmoid(
            self.linear_sum(self.W[0], fc_input, self.U[0], prev_hidden, self.b[0]))
        # reset gate
        r_t = tf.sigmoid(
            self.linear_sum(self.W[1], fc_input, self.U[1], prev_hidden,  self.b[1]))

        # hidden state
        h_t_1 = (1 - u_t) * prev_hidden
        h_t_2 = u_t * tf.tanh(self.linear_sum(self.W[2], fc_input,
                                              self.U[2], r_t * prev_hidden, self.b[2]))
        h_t = h_t_1 + h_t_2
        return h_t


class LSTM_Grid:
    def __init__(self, initializer=None, N=4, n_cells=4, n_input=1024, n_hidden_state=128):
        with tf.name_scope("LSTM_Grid"):
            self.N = 4
            self.n_cells = 4
            self.n_input = 1024
            self.n_hidden_state = 128

            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            self.W = [Weight_Matrix_Grid(initializer=init)]*N
            self.U = [tf.Variable(init(
                [3, 3, 3, n_hidden_state, n_hidden_state]), name="U")]*N
            self.b = [tf.Variable(init(
                [n_cells, n_cells, n_cells, n_hidden_state]), name="b")]*N

    def linear_sum(self, W, x, U, h, b):
        return W.multiply_grid(x) + tf.nn.conv3d(h, U, strides=[1, 1, 1, 1, 1], padding="SAME") + b

    def call(self, fc_input, prev_state):
        prev_hidden_state, prev_cell_state = prev_state

        # forget gate
        f_t = tf.sigmoid(
            self.linear_sum(self.W[0], fc_input, self.U[0], prev_hidden_state, self.b[0]))

        # input gate
        i_t = tf.sigmoid(
            self.linear_sum(self.W[1], fc_input, self.U[1], prev_hidden_state,  self.b[1]))

        # output gate
        o_t = tf.sigmoid(
            self.linear_sum(self.W[2], fc_input, self.U[2], prev_hidden_state, self.b[2]))

        # memory state
        s_t_1 = f_t * prev_cell_state
        s_t_2 = i_t * tf.tanh(self.linear_sum(self.W[3], fc_input,
                                              self.U[3], prev_hidden_state, self.b[3]))
        s_t = s_t_1 + s_t_2
        h_t = o_t*tf.tanh(s_t)

        return (h_t, s_t)


class Weight_Matrix_Grid:
    def __init__(self,  n_x=1024, n_h=128, n_cells=4, initializer=None):
        with tf.name_scope("Weight_Matrix_Grid"):
            # class variables
            self.n_x = n_x
            self.n_h = n_h
            self.n_cells = n_cells

            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            x_list = []
            for i in range(self.n_cells):
                y_list = []
                for j in range(self.n_cells):
                    z_list = []
                    for k in range(self.n_cells):
                        w_name = "W_{}{}{}".format(i, j, k)
                        W = tf.Variable(init(
                            [self.n_x, self.n_h]), name=w_name)
                        tf.summary.histogram(w_name, W)

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
