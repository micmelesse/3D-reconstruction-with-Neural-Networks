import os
import sys
import math
import numpy as np
import tensorflow as tf
from lib import utils


class GRU_Grid:
    def __init__(self, N=3, n_cells=4, n_input=1024, n_hidden_state=128):
        with tf.name_scope("GRU_Grid"):
            self.N = 3
            self.n_cells = 4
            self.n_input = 1024
            self.n_hidden_state = 128

            xavier = tf.contrib.layers.xavier_initializer()
            self.W = [Weight_Matrix_Grid(initalizer=xavier)]*N
            self.U = [tf.Variable(xavier(
                [3, 3, 3, n_hidden_state, n_hidden_state]), name="U")]*N
            self.b = [tf.Variable(xavier(
                [n_cells, n_cells, n_cells, n_hidden_state]), name="b")]*N

    def linear_sum(self, W, x, U, h, b):
        return W.multiply(x) + tf.nn.conv3d(h, U, strides=[1, 1, 1, 1, 1], padding="SAME") + b

    def call(self, fc_input, prev_hidden):
        if prev_hidden is None:
            prev_hidden = tf.zeros(
                [1, self.n_cells, self.n_cells, self.n_cells,  self.n_hidden_state])

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
    def __init__(self, N=4, n_cells=4, n_input=1024, n_hidden_state=128):
        with tf.name_scope("LSTM_Grid"):
            self.N = 4
            self.n_cells = 4
            self.n_input = 1024
            self.n_hidden_state = 128

            xavier = tf.contrib.layers.xavier_initializer()
            self.W = [Weight_Matrix_Grid(initalizer=xavier)]*N
            self.U = [tf.Variable(xavier(
                [3, 3, 3, n_hidden_state, n_hidden_state]), name="U")]*N
            self.b = [tf.Variable(xavier(
                [n_cells, n_cells, n_cells, n_hidden_state]), name="b")]*N

    def linear_sum(self, W, x, U, h, b):
        return W.multiply(x) + tf.nn.conv3d(h, U, strides=[1, 1, 1, 1, 1], padding="SAME") + b

    def call(self, fc_input, state_tuple):
        if state_tuple is None:
            prev_memory, prev_hidden = tf.zeros(
                [1, self.n_cells, self.n_cells, self.n_cells,  self.n_hidden_state]), tf.zeros(
                [1, self.n_cells, self.n_cells, self.n_cells,  self.n_hidden_state])

        prev_memory, prev_hidden = state_tuple

        # forget gate
        f_t = tf.sigmoid(
            self.linear_sum(self.W[0], fc_input, self.U[0], prev_hidden, self.b[0]))

        # input gate
        i_t = tf.sigmoid(
            self.linear_sum(self.W[1], fc_input, self.U[1], prev_hidden,  self.b[1]))

        # output gate
        o_t = tf.sigmoid(
            self.linear_sum(self.W[2], fc_input, self.U[2], prev_hidden, self.b[2]))

        # memory state
        s_t_1 = f_t * prev_memory
        s_t_2 = i_t * tf.tanh(self.linear_sum(self.W[3], fc_input,
                                              self.U[3], prev_hidden, self.b[3]))
        s_t = s_t_1 + s_t_2
        h_t = o_t*tf.tanh(s_t)

        return (s_t, h_t)


class Weight_Matrix_Grid:
    def __init__(self,  n_x=1024, n_h=128, n_cells=4, initalizer=tf.random_normal):
        with tf.name_scope("Weight_Matrix_Grid"):
            # class variables
            self.n_x = n_x
            self.n_h = n_h
            self.n_cells = n_cells
            self.initalizer = initalizer
            x_list = []
            for i in range(self.n_cells):
                y_list = []
                for j in range(self.n_cells):
                    z_list = []
                    for k in range(self.n_cells):
                        weight_matrix = tf.Variable(self.initalizer(
                            [self.n_x, self.n_h]), name="W_{}{}{}".format(i, j, k))  # added on more wieght vector as the bias
                        z_list.append(weight_matrix)
                    y_list.append(z_list)
                x_list.append(y_list)
            self.weight_matrix_grid = x_list

    # multiply each of weight matrix with x
    def multiply(self, x):
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
