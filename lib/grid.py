import math
import numpy as np
import tensorflow as tf


def bias_grid(name, N, n_h):
    i_list = []
    for i in range(N):
        j_list = []
        for j in range(N):
            k_list = []
            for k in range(N):
                k_list.append(tf.Variable(
                    tf.fill([n_h], 1 / math.sqrt(1024)), name="W{}_{}{}{}".format(name, i, j, k)))
            j_list.append(k_list)
        i_list.append(j_list)

    return i_list


def weight_grid(name, N, n_x, n_h):
    i_list = []
    for i in range(N):
        j_list = []
        for j in range(N):
            k_list = []
            for k in range(N):
                k_list.append(tf.Variable(tf.fill(
                    [n_x, n_h], 1 / math.sqrt(n_x)), name="W{}_{}{}{}".format(name, i, j, k)))
            j_list.append(k_list)
        i_list.append(j_list)

    return i_list


def weight_grid_multiply(x, W, N=4):
    W = np.array(W)
    i_list = []
    for i in range(N):
        j_list = []
        for j in range(N):
            k_list = []
            for k in range(N):
                k_list.append(tf.matmul(x, W[i, j, k]))
            j_list.append(k_list)
        i_list.append(j_list)
    return tf.transpose(tf.convert_to_tensor(i_list), [3, 0, 1, 2, 4])


def grid3D(element, N=4):
    return np.array([[[element for k in range(N)] for j in range(N)] for i in range(N)])

# returns the neighboors of a cell including the cell


def get_neighbors(grid, loc=(0, 0, 0), dist=1):
    i, j, k = loc[0], loc[1], loc[2]
    i_min, i_max = max(
        0, i - dist), min(grid.shape[0] - 1, i + dist)
    j_min, j_max = max(
        0, j - dist), min(grid.shape[1] - 1, j + dist)
    k_min, k_max = max(
        0, k - dist), min(grid.shape[1] - 1, k + dist)

    # return np.delete(grid[i_min:i_max + 1, j_min:j_max + 1, k_min:k_max + 1].flatten(), loc)
    return (grid[i_min:i_max + 1, j_min:j_max + 1, k_min:k_max + 1]).flatten()
