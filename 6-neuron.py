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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        m = X.shape[1]
        self.__W = self.__W - (alpha * (np.dot(A - Y, X.T) / m))
        self.__b = self.__b - (alpha * np.sum((A - Y)) / m)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        if not(isinstance(iterations, int)):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not(isinstance(alpha, float)):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be  positive")
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
        return self.evaluate(X, Y)
