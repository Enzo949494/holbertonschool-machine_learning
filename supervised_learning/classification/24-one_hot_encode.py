#!/usr/bin/env python3
"""Function to convert numeric labels to one-hot encoding"""

import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix

    Args:
        Y: numpy.ndarray with shape (m,) containing numeric class labels
           m is the number of examples
        classes: the maximum number of classes found in Y

    Returns:
        A one-hot encoding of Y with shape (classes, m), or None on failure
    """
    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or classes <= 0:
        return None
    if Y.size == 0:
        return None
    if np.any(Y < 0) or np.any(Y >= classes):
        return None

    m = Y.shape[0]

    # Create one-hot matrix with shape (classes, m)
    one_hot = np.zeros((classes, m))

    # Set appropriate positions to 1
    # Y[i] gives the class for example i
    # So we set one_hot[Y[i], i] = 1
    one_hot[Y, np.arange(m)] = 1

    return one_hot
