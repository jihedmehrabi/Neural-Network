import numpy as np


class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        if not isinstance(layers[0], int) or layers[0] <= 0:
            raise TypeError("layers must be a list of positive integers")
        self.weights = {f"W{1}": np.random.randn(layers[0], nx) * np.sqrt(2 / nx),
                        f"b{1}": np.zeros(layers[0], 1)}
        for i in range(1, self.L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            self.weights[f"W{i + 1}"] = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.weights[f"b{i + 1}"] = np.zeros(layers[i], 1)
