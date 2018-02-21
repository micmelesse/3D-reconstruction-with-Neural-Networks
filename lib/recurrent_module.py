import os
import sys
import math
import numpy as np
import tensorflow as tf
import lib.network as network


class GRU_GRID:
    def __init__(self, N=3, n_cells=4, n_input=1024, n_hidden_state=128):
        self.N = 3
        self.n_cells = 4
        self.n_input = 1024
        self.n_hidden_state = 128

        self.W = [WEIGHT_MATRIX_GRID()]*N
        self.U = [tf.Variable(tf.random_normal(
            [3, 3, 3, n_hidden_state, n_hidden_state]))]*N

    def call(self, fc_input, prev_state):
        if prev_state is None:
            prev_state = tf.zeros(
                [1, self.n_cells, self.n_cells, self.n_cells,  self.n_hidden_state])

        Wx_0 = self.W[0].multiply(fc_input)
        Uh_0 = tf.nn.conv3d(prev_state, self.U[0], strides=[
                            1, 1, 1, 1, 1], padding="SAME")

        update_gate = tf.sigmoid(Wx_0+Uh_0)

        return update_gate


class WEIGHT_MATRIX_GRID:
    def __init__(self,  n_x=1024, n_h=128, n_cells=4, grid_initalizer=tf.random_normal):
        # class variables
        self.n_x = n_x
        self.n_h = n_h
        self.n_cells = n_cells
        self.grid_initalizer = grid_initalizer

        def create_weight_matrix_grid():
            x_list = []
            for i in range(self.n_cells):
                y_list = []
                for j in range(self.n_cells):
                    z_list = []
                    for k in range(self.n_cells):
                        grid_index = "{}{}{}".format(i, j, k)
                        weight_matrix = tf.Variable(self.grid_initalizer(
                            [self.n_x+1, self.n_h]), name=grid_index)  # added on more wieght vector as the bias
                        z_list.append(weight_matrix)
                    y_list.append(z_list)
                x_list.append(y_list)
            return x_list

        self.weight_matrix_grid = create_weight_matrix_grid()

    def multiply(self, x):
        print(x.shape)
        x = tf.concat([1, x], 0)
        print(x.shape)

        # multiply each weight matrix with x
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


class GRU_TENSOR:  # use vars that donot fit into memory
    def __init__(self):
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

    def call(self, fc_input, prev_state):
        if prev_state is None:
            prev_state = tf.zeros(
                [1, self.n_cells, self.n_cells, self.n_cells,  self.n_hidden_state])

        fc_input = r2n2_stack(fc_input)
        u_t = tf.sigmoid(
            r2n2_linear(fc_input, self.W[0], self.U[0], prev_state, self.b[0]))
        r_t = tf.sigmoid(
            r2n2_linear(fc_input, self.W[1], self.U[1], prev_state,  self.b[1]))
        h_t = tf.multiply(1 - u_t, prev_state) + tf.multiply(u_t, tf.tanh(
            r2n2_linear(fc_input, self.W[2], self.U[2], tf.multiply(r_t, prev_state), self.b[2])))

        return h_t


class LSTM_GRID:
    def __init__(self, batch_size):
        state_shape = [4, 4, 4, batch_size, 256]
        self.state = tf.contrib.rnn.LSTMStateTuple(
            tf.fill(state_shape, tf.to_float(0)), tf.fill(state_shape, tf.to_float(0)))
        self.W_f = tf.Variable(tf.ones([4, 4, 4, 1024, 256]), name="W_f")
        self.W_i = tf.Variable(tf.ones([4, 4, 4, 1024, 256]), name="W_i")
        self.W_o = tf.Variable(tf.ones([4, 4, 4, 1024, 256]), name="W_o")
        self.W_c = tf.Variable(tf.ones([4, 4, 4, 1024, 256]), name="W_c")

        self.U_f = tf.Variable(tf.ones([3, 3, 3, 256, 256]), name="U_f")
        self.U_i = tf.Variable(tf.ones([3, 3, 3, 256, 256]), name="U_i")
        self.U_o = tf.Variable(tf.ones([3, 3, 3, 256, 256]), name="U_o")
        self.U_c = tf.Variable(tf.ones([3, 3, 3, 256, 256]), name="U_c")

        self.b_f = tf.Variable(tf.ones([4, 4, 4, 1, 256]), name="b_f")
        self.b_i = tf.Variable(tf.ones([4, 4, 4, 1, 256]), name="b_i")
        self.b_o = tf.Variable(tf.ones([4, 4, 4, 1, 256]), name="b_o")
        self.b_c = tf.Variable(tf.ones([4, 4, 4, 1, 256]), name="b_c")

    def call(self, fc_input, prev_state_tuple):
        def linear(x, W, U, b):
            hidden_state, _ = prev_state_tuple
            Wx = tf.matmul(x, W)
            Uh = tf.nn.conv3d(hidden_state, U, strides=[
                1, 1, 1, 1, 1], padding="SAME")
            print(Wx.shape, Uh.shape, b.shape)
            return Wx + Uh + b

        _, prev_output = prev_state_tuple
        f_t = tf.sigmoid(linear(fc_input, self.W_f, self.U_f, self.b_f))
        i_t = tf.sigmoid(linear(fc_input, self.W_i, self.U_i, self.b_i))
        o_t = tf.sigmoid(linear(fc_input, self.W_o, self.U_o, self.b_o))
        c_t = tf.multiply(f_t, prev_output) + tf.multiply(i_t,
                                                          tf.tanh(linear(fc_input, self.W_c, self.U_c, self.b_c)))
        h_t = tf.multiply(o_t, tf.tanh(c_t))

        return h_t, tf.contrib.rnn.LSTMStateTuple(c_t, h_t)
