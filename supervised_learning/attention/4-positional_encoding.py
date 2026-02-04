#!/usr/bin/env python3
"""
Positional Encoding Module

Module implements the positional encoding mechanism used in transformer model
to add info about the relative or absolute position of tokens in the sequence.
Positional encoding use sinusoidal functions of different frequencies to encode
position information.
"""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculate positional encoding for transformer attention mechanism.

    Uses sinusoidal positional encoding where even dimensions use sine and odd
    dimensions use cosine functions with different frequencies.

    Args:
        max_seq_len (int): Maximum length of the input sequences
        dm (int): Model dimension (embedding dimension)

    Returns:
        numpy.ndarray: Positional encoding matrix of shape (max_seq_len, dm)
                      containing the positional encodings for each position
    """
    # Positions: (max_seq_len, 1)
    position = np.arange(max_seq_len)[:, np.newaxis]

    # Dimensions: (1, dm)
    div_term = np.exp(np.arange(0, dm, 2) * -(np.log(10000.0) / dm))

    # Angles: (max_seq_len, dm/2)
    pe = np.zeros((max_seq_len, dm))
    pe[:, 0:dm:2] = np.sin(position * div_term)  # sin sur pairs
    pe[:, 1:dm:2] = np.cos(position * div_term)  # cos sur impairs

    return pe
