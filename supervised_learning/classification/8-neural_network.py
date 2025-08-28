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
            
        # Initialize weights for hidden layer using random normal distribution
        self.W1 = np.random.normal(size=(nodes, nx))
        # Initialize bias for hidden layer with zeros
        self.b1 = np.zeros((nodes, 1))
        # Initialize activated output for hidden layer to 0
        self.A1 = 0
        
        # Initialize weights for output neuron using random normal distribution
        self.W2 = np.random.normal(size=(1, nodes))
        # Initialize bias for output neuron to 0
        self.b2 = 0
        # Initialize activated output for output neuron (prediction) to 0
        self.A2 = 0