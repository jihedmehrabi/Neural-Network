import numpy as np
import matplotlib.pyplot as plt


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

    def gradient_descent(self, Y, cache, alpha=0.05):
        m = cache["A0"].shape[1]
        A = cache[f"A{self.__L}"]
        tempw = {}
        tempz = {}
        tempb = {}
        tempz[f"dz{self.__L}"] = A - Y
        tempw[f"dw{self.__L}"] = np.matmul(tempz[f"dz{self.__L}"], cache[f"A{self.__L-1}"].T) / m
        tempb[f"db{self.__L}"] = np.sum(tempz[f"dz{self.__L}"], axis=1, keepdims=True) / m

        for i in range(self.__L - 1, 0, -1):
            A1 = cache[f"A{i}"]
            A0 = cache[f"A{i-1}"]
            tempz[f"dz{i}"] = np.matmul(self.__weights[f"W{i+1}"].T, tempz[f"dz{i+1}"]) * (A1 * (1 - A1))
            tempw[f"dw{i}"] = np.matmul(tempz[f"dz{i}"], A0.T) / m
            tempb[f"db{i}"] = np.sum(tempz[f"dz{i}"], axis=1, keepdims=True) / m
            self.__weights[f"W{i+1}"] -= alpha * tempw[f"dw{i+1}"]
            self.__weights[f"b{i+1}"] -= alpha * tempb[f"db{i+1}"]
        self.__weights["W1"] -= alpha * tempw["dw1"]
        self.__weights["b1"] -= alpha * tempb["db1"]

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        if not(isinstance(iterations, int)):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not(isinstance(alpha, float)):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be  positive")
        if verbose or graph:
            if not(isinstance(step, int)):
                raise TypeError('step must be an integer')
            if step <= 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')
        costs = []
        iteration_s = []
        for iteration in range(iterations + 1):
            self.forward_prop(X)
            A = self.__cache[f"A{self.__L}"]
            cost = self.cost(Y, A)
            if verbose and iteration % step == 0:
                print(f'Cost after {iteration} iterations: {cost}')
            if graph and iteration % step == 0:
                costs.append(cost)
                iteration_s.append(iteration)
            self.gradient_descent(Y, self.__cache, alpha)
        if graph:
            plt.plot(iteration_s, costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("training cost")
            plt.show()
        return self.evaluate(X, Y)
