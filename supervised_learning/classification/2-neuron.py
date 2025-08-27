#!/usr/bin/env python3
"""Module for Neuron class"""

import numpy as np


class Neuron:
    """Class that defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """
        Constructor for Neuron class

        Args:
            nx: number of input features to the neuron

        Raises:
            TypeError: if nx is not an integer
            ValueError: if nx is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Initialize private weights using random normal distribution
        self.__W = np.random.normal(size=(1, nx))
        # Initialize private bias to 0
        self.__b = 0
        # Initialize private activated output to 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights"""
        return self.__W

    @property
    def b(self):
        """Getter for bias"""
        return self.__b

    @property
    def A(self):
        """Getter for activated output"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron

        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
               nx is the number of input features to the neuron
               m is the number of examples

        Returns:
            The activated output of the neuron (self.__A)
        """
        # Calculate the linear transformation: z = W * X + b
        z = np.dot(self.__W, X) + self.__b

        # Apply sigmoid activation function: A = 1 / (1 + e^(-z))
        self.__A = 1 / (1 + np.exp(-z))

        return self.__A
