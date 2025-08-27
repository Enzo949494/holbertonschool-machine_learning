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

        # Initialize weights using random normal distribution
        self.W = np.random.normal(size=(1, nx))
        # Initialize bias to 0
        self.b = 0
        # Initialize activated output to 0
        self.A = 0
