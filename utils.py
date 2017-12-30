import math
import numpy as np
import tensorflow as tf
from PIL import Image


def main():
    pass


def imshow_multichannel(im):
    n_channels = im.shape[-1]
    n_tile = math.floor(math.sqrt(n_channels))

    rows = []
    for i in range(0, n_channels, n_tile):
        if(i + n_tile < n_channels):
            rows.append(np.concatenate(im[:, :, i:i + n_tile],axis=1))
        else:
            padding = np.concatenate(np.zeros([im.shape[0],im.shape[1],i + n_tile - n_channels]),axis=1)
            data=np.concatenate(im[:, :, i:n_channels],axis=1)
            rows.append(np.concatenate((data,padding),axis=1))
    return rows


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
