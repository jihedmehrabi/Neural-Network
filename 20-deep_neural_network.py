import numpy as np


class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        if not isinstance(layers[0], int) or layers[0] <= 0:
            raise TypeError("layers must be a list of positive integers")
        self.__weights = {f"W{1}": np.random.randn(layers[0], nx) * np.sqrt(2 / nx),
                          f"b{1}": np.zeros((layers[0], 1))}
        for i in range(1, self.L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            self.__weights[f"W{i + 1}"] = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.__weights[f"b{i + 1}"] = np.zeros((layers[i], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        self.__cache["A0"] = X
        for i in range(1, self.__L + 1):
            Z = np.dot(self.__weights[f"W{i}"], self.__cache[f"A{i-1}"]) + self.__weights[f"b{i}"]
            self.__cache[f"A{i}"] = 1 / (1 + np.exp(-Z))
        return self.__cache["A3"], self.__cache

    def cost(self, Y, A):
        m = Y.shape[1]
        xmp = (1 - Y)
        cost = -1 / m * (np.sum(Y * np.log(A) + xmp * np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        self.forward_prop(X)
        A = self.__cache[f"A{self.__L}"]
        cost = self.cost(Y, A)
        pred = np.where(A < 0.5, 0, 1)
        return pred, cost
