#!/usr/bin/env python3
"""
GRU cell
"""

import numpy as np


class GRUCell:
    """
    Represents a cell of a Gated Recurrent Unit (GRU).
    """

    def __init__(self, i, h, o):
        """
        Initialize a GRU cell.

        Args:
            i (int): Dimensionality of the input data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the outputs.
        """
        # Poids pour les portes (entrée x_t et état caché h_prev séparés)
        self.Wz = np.random.randn(i, h)
        self.Wr = np.random.randn(i, h)
        self.Wh = np.random.randn(i, h)

        self.Uz = np.random.randn(h, h)
        self.Ur = np.random.randn(h, h)
        self.Uh = np.random.randn(h, h)

        # Poids pour la sortie
        self.Wy = np.random.randn(h, o)

        # Bias
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): Previous hidden state, shape (m, h).
            x_t (numpy.ndarray): Input at time t, shape (m, i).

        Returns:
            h_next (numpy.ndarray): Next hidden state, shape (m, h).
            y (numpy.ndarray): Output at time t, shape (m, o).
        """
        # Update gate
        z = 1 / (1 + np.exp(-(x_t @ self.Wz + h_prev @ self.Uz + self.bz)))

        # Reset gate
        r = 1 / (1 + np.exp(-(x_t @ self.Wr + h_prev @ self.Ur + self.br)))

        # Candidate hidden state
        h_tilde = np.tanh(
            x_t @ self.Wh + (r * h_prev) @ self.Uh + self.bh
        )

        # New hidden state
        h_next = (1 - z) * h_prev + z * h_tilde

        # Output with softmax
        y_lin = h_next @ self.Wy + self.by
        exp_y = np.exp(y_lin)
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, y
