#!/usr/bin/env python3

import numpy as np


class RNNCell:
    """Represents a cell of a simple RNN"""

    def __init__(self, i, h, o):
        """
        Initialize RNNCell

        Args:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step

        Args:
            h_prev: numpy.ndarray of shape (m, h) containing the previous hidden state
            x_t: numpy.ndarray of shape (m, i) containing the data input for the cell

        Returns:
            h_next: the next hidden state
            y: the output of the cell
        """
        # Concatenate input and previous hidden state
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Calculate next hidden state with tanh activation
        h_next = np.tanh(np.dot(concat, self.Wh) + self.bh)

        # Calculate output with softmax activation
        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, y