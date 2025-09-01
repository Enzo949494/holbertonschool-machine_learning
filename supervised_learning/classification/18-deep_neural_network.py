#!/usr/bin/env python3
"""Module for DeepNeuralNetwork class"""

import numpy as np


class DeepNeuralNetwork:
    """Class that defines a deep neural network
       performing binary classification"""

    def __init__(self, nx, layers):
        """
        Constructor for DeepNeuralNetwork class

        Args:
            nx: number of input features
            layers: list representing the number of nodes in each layer

        Raises:
            TypeError: if nx is not an integer
            ValueError: if nx is less than 1
            TypeError: if layers is not a list
            TypeError: if elements in layers are not positive integers
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Une seule boucle qui fait validation ET initialisation
        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            if i == 0:
                prev_layer_size = nx
            else:
                prev_layer_size = layers[i - 1]

            current_layer_size = layers[i]

            # Initialize weights using He et al. method
            scale = np.sqrt(2 / prev_layer_size)
            weights = np.random.randn(current_layer_size, prev_layer_size)
            self.__weights[f"W{i + 1}"] = weights * scale

            # Initialize biases to zeros
            self.__weights[f"b{i + 1}"] = np.zeros((current_layer_size, 1))

    @property
    def L(self):
        """Getter for number of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache dictionary"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights dictionary"""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network

        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
               nx is the number of input features to the neuron
               m is the number of examples

        Returns:
            The output of the neural network and the cache, respectively
        """
        # Save input to cache as A0
        self.__cache['A0'] = X

        # Forward propagation through all layers
        A = X
        for i in range(1, self.__L + 1):
            # Get weights and biases for current layer
            W = self.__weights[f'W{i}']
            b = self.__weights[f'b{i}']

            # Calculate linear transformation: Z = W * A + b
            Z = np.dot(W, A) + b

            # Apply sigmoid activation function: A = 1 / (1 + e^(-Z))
            A = 1 / (1 + np.exp(-Z))

            # Save activated output to cache
            self.__cache[f'A{i}'] = A

        return A, self.__cache