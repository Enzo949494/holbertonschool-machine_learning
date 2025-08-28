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
            
        # Initialize weights and biases
        self.W1 = np.random.normal(size=(nodes, nx))
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.normal(size=(1, nodes))
        self.b2 = 0
        self.A2 = 0