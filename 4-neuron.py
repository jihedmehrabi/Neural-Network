import numpy as np


class Neuron:

    def __init__(self, nx=784):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        m = Y.shape[1]
        xmp = (1 - Y)
        cost = -1 / m * np.sum(Y * np.log(A) + xmp * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        pred = np.where(self.__A < 0.5, 0, 1)
        return pred, cost
