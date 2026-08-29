import numpy as np


def one_hot_encode(Y, classes):
    m = Y.shape[0]
    M = np.zeros((classes, m))
    for i in range(classes):
        M[Y[i]][i] = 1
    return M
