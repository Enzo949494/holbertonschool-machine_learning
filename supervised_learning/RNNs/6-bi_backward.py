#!/usr/bin/env python3
"""Bidirectional RNN cell (forward + backward)"""

import numpy as np


class BidirectionalCell:
    """Represents a bidirectional cell of an RNN."""

    def __init__(self, i, h, o):
        """
        i: dimension of the data
        h: dimension of the hidden state
        o: dimension of the outputs
        """
        # Weights for forward hidden state
        self.Whf = np.random.randn(i + h, h)
        # Weights for backward hidden state
        self.Whb = np.random.randn(i + h, h)
        # Weights for outputs (will use [h_f, h_b])
        self.Wy = np.random.randn(2 * h, o)

        # Biases
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Forward direction hidden state for one time step.

        h_prev: (m, h)
        x_t:    (m, i)

        Returns:
            h_next: (m, h)
        """
        concat = np.concatenate((h_prev, x_t), axis=1)  # (m, h+i)
        h_next = np.tanh(concat @ self.Whf + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
        Backward direction hidden state for one time step.

        h_next: (m, h) next hidden state (in backward pass)
        x_t:    (m, i) input at this time step

        Returns:
            h_prev: (m, h) previous hidden state (backward)
        """
        concat = np.concatenate((h_next, x_t), axis=1)  # (m, h+i)
        h_prev = np.tanh(concat @ self.Whb + self.bhb)
        return h_prev
