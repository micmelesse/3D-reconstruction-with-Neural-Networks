import numpy as np


def grid3D(element, N=32):
    return np.array([[[element for k in range(N)] for j in range(N)] for i in range(N)])


def get_neighbors(grid, loc, dist=1):
    i, j, k = loc[0], loc[1], loc[2]
    i_min, i_max = max(
        0, i - dist), min(grid.shape[0] - 1, i + dist)
    j_min, j_max = max(
        0, j - dist), min(grid.shape[1] - 1, j + dist)
    k_min, k_max = max(
        0, k - dist), min(grid.shape[1] - 1, k + dist)

    return np.delete(grid[i_min:i_max + 1, j_min:j_max + 1, k_min:k_max + 1].flatten(), loc)
