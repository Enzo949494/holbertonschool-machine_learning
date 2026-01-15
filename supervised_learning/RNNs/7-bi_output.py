#!/usr/bin/env python3
"""Bidirectional RNN cell (forward, backward, output)"""

import numpy as np


class BidirectionalCell:
    """Represents a bidirectional cell of an RNN."""

    def __init__(self, i, h, o):
        """
        i: dimension of the data
        h: dimension of the hidden state
        o: dimension of the outputs
        """
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)

        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(concat @ self.Whf + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        concat = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(concat @ self.Whb + self.bhb)
        return h_prev

    def output(self, H):
        """
        Calculates all outputs for the RNN.

        H: (t, m, 2*h) concatenated hidden states (forward + backward)

        Returns:
            Y: (t, m, o) outputs after softmax
        """
        t, m, _ = H.shape
        o = self.Wy.shape[1]

        # Affine transformation: (t, m, 2h) -> (t, m, o)
        Y_lin = H @ self.Wy + self.by  # broadcasting by over (t, m)

        # Softmax over last dimension
        exp_Y = np.exp(Y_lin)
        Y = exp_Y / np.sum(exp_Y, axis=2, keepdims=True)

        return Y
