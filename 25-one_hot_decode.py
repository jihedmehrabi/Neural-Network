import numpy as np


def one_hot_decode(one_hot):
    Y = np.array([])
    for i in range(one_hot.shape[1]):
        for j in range(one_hot.shape[0]):
            if one_hot[j][i] == 1:
                Y = np.append(Y, j)
    return Y.astype(int)
