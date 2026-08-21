import numpy as np


class Neuron:

    def __init__(self, nx=784):
        # 1. Vérification du type
        if type(nx) is not int:
            raise TypeError("nx must be an integer")

        # 2. Vérification de la valeur
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # 3. Initialisation des attributs
        self.W = np.random.randn(1, nx)  # Matrice de taille (1, nx)
        self.b = 0  # Biais initialisé à 0
        self.A = 0  # Activation initialisée à 0
