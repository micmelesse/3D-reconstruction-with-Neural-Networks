import os
import sys
import math
import numpy as np
import tensorflow as tf
import lib.utils as utils


class WEIGHT_MATRIX_GRID:
    def __init__(self,  n_x=1024, n_h=128, n_cells=4, initalizer=tf.random_normal):
        # class variables
        self.n_x = n_x
        self.n_h = n_h
        self.n_cells = n_cells
        self.initalizer = initalizer

        with tf.name_scope("WEIGHT_MATRIX_GRID"):
            x_list = []
            for i in range(self.n_cells):
                y_list = []
                for j in range(self.n_cells):
                    z_list = []
                    for k in range(self.n_cells):
                        grid_index = "{}{}{}".format(i, j, k)
                        weight_matrix = tf.Variable(self.initalizer(
                            [self.n_x, self.n_h]), name=grid_index)  # added on more wieght vector as the bias
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


class GRU_GRID:  # GOOD
    def __init__(self, N=3, n_cells=4, n_input=1024, n_hidden_state=128):
        self.N = 3
        self.n_cells = 4
        self.n_input = 1024
        self.n_hidden_state = 128

        gru_initializer = tf.contrib.layers.xavier_initializer()
        self.W = [WEIGHT_MATRIX_GRID(initalizer=gru_initializer)]*N
        self.U = [tf.Variable(gru_initializer(
            [3, 3, 3, n_hidden_state, n_hidden_state]))]*N
        self.b = [tf.Variable(gru_initializer(
            [n_cells, n_cells, n_cells, n_hidden_state]))]*N

    def linear_sum(self, W, x, U, h, b):
        return W.multiply(x) + tf.nn.conv3d(h, U, strides=[1, 1, 1, 1, 1], padding="SAME") + b

    def call(self, fc_input, prev_state):
        if prev_state is None:
            prev_state = tf.zeros(
                [1, self.n_cells, self.n_cells, self.n_cells,  self.n_hidden_state])

        # update gate
        u_t = tf.sigmoid(
            self.linear_sum(self.W[0], fc_input, self.U[0], prev_state, self.b[0]))
        # reset gate
        r_t = tf.sigmoid(
            self.linear_sum(self.W[1], fc_input, self.U[1], prev_state,  self.b[1]))

        # hidden state
        h_t_1 = (1 - u_t) * prev_state
        h_t_2 = u_t * tf.tanh(self.linear_sum(self.W[2], fc_input,
                                              self.U[2], r_t * prev_state, self.b[2]))

        return h_t_1 + h_t_2


class GRU_TENSOR:  # BAD, creates really big tensors that donot fit into gpus
    def __init__(self, N=3, n_cells=4, n_input=1024, n_hidden_state=128):
        self.N = 3
        self.n_cells = 4
        self.n_input = 1024
        self.n_hidden_state = 128

        gru_initializer = tf.contrib.layers.xavier_initializer()
        self.W = [tf.Variable(gru_initializer(
            [self.n_cells,  self.n_cells,  self.n_cells,  self.n_input,  self.n_hidden_state]), name="W_GRU")]*self.N
        self.b = [tf.Variable(gru_initializer(
            [self.n_cells,  self.n_cells,  self.n_cells,  self.n_hidden_state]), name="b_GRU")]*self.N
        self.U = [tf.Variable(gru_initializer(
            [3, 3, 3,  self.n_hidden_state,  self.n_hidden_state]), name="U_GRU")]*self.N

    def linear_sum(self, x, W, U, h, b):
        Wx = tf.map_fn(lambda a: tf.squeeze(
            tf.matmul(tf.expand_dims(a, axis=-2), W), axis=-2), x)
        Uh = tf.nn.conv3d(h, U, strides=[1, 1, 1, 1, 1], padding="SAME")
        return Wx + Uh + b

    def call(self, fc_input, prev_state):
        if prev_state is None:
            prev_state = tf.zeros(
                [1, self.n_cells, self.n_cells, self.n_cells,  self.n_hidden_state])

        # stack input in 4x4x4 tensors
        fc_input = tf.transpose(
            tf.stack([tf.stack([tf.stack([fc_input] * self.n_cells)] * self.n_cells)] * self.n_cells), [3, 0, 1, 2, 4])

        # update gate
        u_t = tf.sigmoid(
            self.linear_sum(self.W[0], fc_input, self.U[0], prev_state, self.b[0]))
        # reset gate
        r_t = tf.sigmoid(
            self.linear_sum(self.W[1], fc_input, self.U[1], prev_state,  self.b[1]))

        # hidden state
        h_t_1 = (1 - u_t) * prev_state
        h_t_2 = u_t * tf.tanh(self.linear_sum(self.W[2], fc_input,
                                              self.U[2], r_t * prev_state, self.b[2]))

        return h_t_1 + h_t_2
