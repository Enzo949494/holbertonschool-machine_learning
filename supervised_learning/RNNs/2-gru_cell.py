#!/usr/bin/env python3
"""
Module implementing a Gated Recurrent Unit (GRU) cell.

This module contains the GRUCell class which represents a single cell of a
Gated Recurrent Unit that can process sequential data with update and reset
gates to control information flow.
"""

import numpy as np


class GRUCell:
    """
    Represents a cell of a Gated Recurrent Unit (GRU).

    A GRU cell processes input data and maintains a hidden state across
    time steps using update and reset gates. It uses sigmoid activation
    for the gates, tanh activation for the intermediate hidden state,
    and softmax for the output.

    Attributes:
        Wz (numpy.ndarray): Weight matrix for update gate. Shape: (i + h, h)
        Wr (numpy.ndarray): Weight matrix for reset gate. Shape: (i + h, h)
        Wh (numpy.ndarray): Weight matrix for intermediate hidden state.
                           Shape: (i + h, h)
        Wy (numpy.ndarray): Weight matrix for output. Shape: (h, o)
        bz (numpy.ndarray): Bias for update gate. Shape: (1, h)
        br (numpy.ndarray): Bias for reset gate. Shape: (1, h)
        bh (numpy.ndarray): Bias for intermediate hidden state. Shape: (1, h)
        by (numpy.ndarray): Bias for output. Shape: (1, o)
    """

    def __init__(self, i, h, o):
        """
        Initialize a GRU cell.

        Args:
            i (int): Dimensionality of the input data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the outputs.
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
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
                - y (numpy.ndarray): Output of the cell. Shape: (m, o).
                                    Computed using softmax activation.
        """
        # Concatenate input and previous hidden state
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Update gate
        z = 1 / (1 + np.exp(-(np.dot(concat, self.Wz) + self.bz)))

        # Reset gate
        r = 1 / (1 + np.exp(-(np.dot(concat, self.Wr) + self.br)))

        # Intermediate hidden state
        h_tilde_concat = np.concatenate((r * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.dot(h_tilde_concat, self.Wh) + self.bh)

        # Next hidden state
        h_next = (1 - z) * h_tilde + z * h_prev

        # Output with softmax activation
        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, y