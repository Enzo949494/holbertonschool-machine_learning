#!/usr/bin/env python3
"""
Module implementing forward propagation for a simple RNN.

This module contains the rnn function which performs forward propagation
for a simple Recurrent Neural Network over multiple time steps.
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Perform forward propagation for a simple RNN.

    Args:
        rnn_cell: An instance of RNNCell that will be used for the forward
                 propagation.
        X (numpy.ndarray): Data to be used for forward propagation.
                          Shape: (t, m, i) where:
                          - t is the maximum number of time steps
                          - m is the batch size
                          - i is the dimensionality of the data
        h_0 (numpy.ndarray): Initial hidden state.
                            Shape: (m, h) where:
                            - m is the batch size
                            - h is the dimensionality of the hidden state

    Returns:
        tuple: A tuple containing:
            - H (numpy.ndarray): All hidden states throughout the sequence.
                                Shape: (t + 1, m, h)
                                Includes the initial hidden state h_0 at index 0
            - Y (numpy.ndarray): All outputs throughout the sequence.
                                Shape: (t, m, o) where o is the output
                                dimensionality
    """
    # Get dimensions
    t, m, i = X.shape
    h = h_0.shape[1]
    o = rnn_cell.Wy.shape[1]

    # Initialize arrays to store hidden states and outputs
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))

    # Set initial hidden state
    H[0] = h_0

    # Iterate through time steps
    for step in range(t):
        # Get input for current time step
        x_t = X[step]

        # Forward propagation through RNN cell
        h_next, y = rnn_cell.forward(H[step], x_t)

        # Store hidden state and output
        H[step + 1] = h_next
        Y[step] = y

    return H, Y