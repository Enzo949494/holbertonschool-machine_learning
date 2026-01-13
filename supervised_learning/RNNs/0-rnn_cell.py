#!/usr/bin/env python3
"""
Module implementing a simple RNN cell for recurrent neural networks.

This module contains the RNNCell class which represents a single cell of a
simple Recurrent Neural Network (RNN) that can process sequential data.
"""

import numpy as np


class RNNCell:
    """
    Represents a cell of a simple RNN.

    A RNN cell processes input data and maintains a hidden state across
    time steps. It uses tanh activation for the hidden state and softmax
    for the output.

    Attributes:
        Wh (numpy.ndarray): Weight matrix concatenated hidden state and input.
                           Shape: (i + h, h)
        Wy (numpy.ndarray): Weight matrix for output. Shape: (h, o)
        bh (numpy.ndarray): Bias for hidden state. Shape: (1, h)
        by (numpy.ndarray): Bias for output. Shape: (1, o)
    """

    def __init__(self, i, h, o):
        """
        Initialize a RNN cell.

        Args:
            i (int): Dimensionality of the input data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the outputs.
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): Previous hidden state. Shape: (m, h)
                                   where m is the batch size.
            x_t (numpy.ndarray): Input data for the current time step.
                                Shape: (m, i) where m is the batch size.

        Returns:
            tuple: A tuple containing:
                - h_next (numpy.ndarray): Next hidden state. Shape: (m, h).
                                         Computed using tanh activation.
                - y (numpy.ndarray): Output of the cell. Shape: (m, o).
                                    Computed using softmax activation.
        """
        # Concatenate input and previous hidden state
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Calculate next hidden state with tanh activation
        h_next = np.tanh(np.dot(concat, self.Wh) + self.bh)

        # Calculate output with softmax activation
        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, y
