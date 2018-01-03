import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def main():
    pass

def imshow_multichannel(im):
    # print ("use %matplotlib inline if you want to display result in a notebook")
    plt.figure()
    return plt.imshow(flatten_multichannel_image(im))

def flatten_multichannel_image(im):
    n_channels = im.shape[-1]
    n_tile = math.ceil(math.sqrt(n_channels))
    rows = []
    for i in range(n_tile):
        c = i * n_tile
        if c < n_channels:
            a = im[:, :, c]
            for j in range(1, n_tile):
                c = i * n_tile + j
                if c < n_channels:
                    b = im[:, :, c]
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


def unpool(value, name='unpool'):
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat([out, tf.zeros_like(out)], i)
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out


def grid3D(element, N=2):
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


if __name__ == '__main__':
    main()
