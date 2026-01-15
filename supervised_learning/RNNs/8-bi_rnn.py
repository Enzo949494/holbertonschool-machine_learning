#!/usr/bin/env python3
"""Bidirectional RNN full forward"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN.

    bi_cell: BidirectionalCell instance
    X: np.ndarray (t, m, i) input data
    h_0: np.ndarray (m, h) initial hidden state (forward)
    h_t: np.ndarray (m, h) initial hidden state (backward)

    Returns:
        H: np.ndarray (t, m, 2*h) concatenated hidden states
        Y: np.ndarray (t, m, o) outputs
    """
    t, m, _ = X.shape
    h = h_0.shape[1]

    # Etats cachés forward et backward
    H_forward = np.zeros((t, m, h))
    H_backward = np.zeros((t, m, h))

    # Forward pass
    h_prev = h_0
    for time in range(t):
        x_t = X[time]
        h_prev = bi_cell.forward(h_prev, x_t)
        H_forward[time] = h_prev

    # Backward pass
    h_next = h_t
    for time in reversed(range(t)):
        x_t = X[time]
        h_next = bi_cell.backward(h_next, x_t)
        H_backward[time] = h_next

    # Concaténation des états cachés
    H = np.concatenate((H_forward, H_backward), axis=2)  # (t, m, 2*h)

    # Sorties
    Y = bi_cell.output(H)

    return H, Y
