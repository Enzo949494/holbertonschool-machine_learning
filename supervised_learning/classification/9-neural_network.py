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