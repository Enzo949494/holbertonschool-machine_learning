#!/usr/bin/env python3
"""Bidirectional RNN cell (forward)"""

import numpy as np


class BidirectionalCell:
    """Represents a bidirectional cell of an RNN."""

    def __init__(self, i, h, o):
        """
        i: dimension of the data
        h: dimension of the hidden state
        o: dimension of the outputs
        """
        # Weights for forward direction hidden state
        self.Whf = np.random.randn(i + h, h)
        # Weights for backward direction hidden state
        self.Whb = np.random.randn(i + h, h)
        # Weights for outputs (will use [h_forward, h_backward])
        self.Wy = np.random.randn(2 * h, o)

        # Biases
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculates the hidden state in the forward direction for one time step.

        h_prev: (m, h) previous hidden state (forward)
        x_t:    (m, i) input at time t

        Returns:
            h_next: (m, h) next hidden state (forward)
        """
        # Concatenate previous hidden state and input
        concat = np.concatenate((h_prev, x_t), axis=1)  # (m, h+i)

        # Compute next hidden state with tanh activation
        h_next = np.tanh(concat @ self.Whf + self.bhf)

        return h_next
