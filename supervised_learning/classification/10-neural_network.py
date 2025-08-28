#!/usr/bin/env python3
"""Module for NeuralNetwork class"""

import numpy as np


class NeuralNetwork:
    """Class that defines a neural network with one hidden layer performing binary classification"""

    def __init__(self, nx, nodes):
        """
        Constructor for NeuralNetwork class
        
        Args:
            nx: number of input features
            nodes: number of nodes found in the hidden layer
            
        Raises:
            TypeError: if nx is not an integer
            ValueError: if nx is less than 1
            TypeError: if nodes is not an integer
            ValueError: if nodes is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
            
        # Initialize private attributes
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for weights of hidden layer"""
        return self.__W1

    @property
    def b1(self):
        """Getter for bias of hidden layer"""
        return self.__b1

    @property
    def A1(self):
        """Getter for activated output of hidden layer"""
        return self.__A1

    @property
    def W2(self):
        """Getter for weights of output neuron"""
        return self.__W2

    @property
    def b2(self):
        """Getter for bias of output neuron"""
        return self.__b2

    @property
    def A2(self):
        """Getter for activated output of output neuron"""
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        
        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
               nx is the number of input features to the neuron
               m is the number of examples
               
        Returns:
            The private attributes __A1 and __A2, respectively
        """
        # Calculate the linear transformation for hidden layer: z1 = W1 * X + b1
        z1 = np.dot(self.__W1, X) + self.__b1
        
        # Apply sigmoid activation function to hidden layer: A1 = 1 / (1 + e^(-z1))
        self.__A1 = 1 / (1 + np.exp(-z1))
        
        # Calculate the linear transformation for output layer: z2 = W2 * A1 + b2
        z2 = np.dot(self.__W2, self.__A1) + self.__b2
        
        # Apply sigmoid activation function to output layer: A2 = 1 / (1 + e^(-z2))
        self.__A2 = 1 / (1 + np.exp(-z2))
        
        return self.__A1, self.__A2