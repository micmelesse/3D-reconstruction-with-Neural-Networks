import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def main():
    pass

# print ("use %matplotlib inline if you want to display result in a notebook")
def imsave_multichannel(im, f_name):
    return plt.imsave(f_name, (flatten_multichannel_image(im)))

# print ("use %matplotlib inline if you want to display result in a notebook")
def imshow_multichannel(im):  
    return plt.imshow(flatten_multichannel_image(im))


def flatten_multichannel_image(im):
    #print(im.shape)
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


if __name__ == '__main__':
    main()
