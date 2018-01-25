import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def imsave_multichannel(im, f_name):
    return plt.imsave(f_name, (flatten_multichannel_image(im)))


def imshow_multichannel(im):
    return plt.imshow(flatten_multichannel_image(im))


def flatten_multichannel_image(im):
    # print(im.shape)
    n_channels = im.shape[-1]
    n_tile = math.ceil(math.sqrt(n_channels))
    rows = []
    for i in range(n_tile):
        c_first = i * n_tile
        if c_first < n_channels:
            a = im[:, :, c_first]
            for j in range(1, n_tile):
                c_last = c_first + j
                if c_last < n_channels:
                    b = im[:, :, c_last]
                    a = np.hstack((a, b))
                else:
                    b = np.zeros([im.shape[0], im.shape[1]],)
                    a = np.hstack((a, b))
            rows.append(a)

    n = len(rows)
    a = rows[0]
    for i in range(1, n):
        b = rows[i]
        a = np.vstack((a, b))
    return a


# vis voxels
def imshow_voxel(vox):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(vox, edgecolor='k')
    ax.view_init(30, 30)
    return plt.show()


def imsave_voxel(vox, f_name):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(vox, edgecolor='k')
    ax.view_init(30, 30)
    plt.savefig(f_name, bbox_inches='tight')
    plt.close()


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

# #class Unpool3DLayer(Layer):
#     """3D Unpooling layer for a convolutional network """

#     def __init__(self, prev_layer, unpool_size=(2, 2, 2), padding=(0, 0, 0)):
#         super().__init__(prev_layer)
#         self._unpool_size = unpool_size
#         self._padding = padding
#         output_shape = (self._input_shape[0],  # batch
#                         unpool_size[0] * self._input_shape[1] + 2 * padding[0],  # depth
#                         self._input_shape[2],  # out channel
#                         unpool_size[1] * self._input_shape[3] + 2 * padding[1],  # row
#                         unpool_size[2] * self._input_shape[4] + 2 * padding[2])  # col
#         self._output_shape = output_shape

#     def set_output(self):
#         output_shape = self._output_shape
#         padding = self._padding
#         unpool_size = self._unpool_size
#         unpooled_output = tensor.alloc(0.0,  # Value to fill the tensor
#                                        output_shape[0],
#                                        output_shape[1] + 2 * padding[0],
#                                        output_shape[2],
#                                        output_shape[3] + 2 * padding[1],
#                                        output_shape[4] + 2 * padding[2])

#         unpooled_output = tensor.set_subtensor(unpooled_output[:, padding[0]:output_shape[
#             1] + padding[0]:unpool_size[0], :, padding[1]:output_shape[3] + padding[1]:unpool_size[
#                 1], padding[2]:output_shape[4] + padding[2]:unpool_size[2]],
#                                                self._prev_layer.output)
#         self._output = unpooled_output



def unpool3D(value, name='unpool3D'):
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1: -1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat([out, tf.zeros_like(out)], i)
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out


def stack_grid(x, N=4):
    x = tf.expand_dims(x, axis=1)
    return tf.transpose(tf.stack([tf.stack([tf.stack([x] * N)] * N)] * N), [3, 0, 1, 2, 4, 5])


def grid3D(element, N=4):
    return np.array([[[element for k in range(N)] for j in range(N)] for i in range(N)])
