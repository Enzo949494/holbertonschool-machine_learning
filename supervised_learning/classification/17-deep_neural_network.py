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
