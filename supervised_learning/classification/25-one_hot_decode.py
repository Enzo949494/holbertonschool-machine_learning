#!/usr/bin/env python3
"""Function to convert one-hot encoding back to numeric labels"""

import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels

    Args:
        one_hot: one-hot encoded numpy.ndarray with shape (classes, m)
                 classes is the maximum number of classes
                 m is the number of examples

    Returns:
        A numpy.ndarray with shape (m,) containing the numeric labels
        for each example, or None on failure
    """
    if not isinstance(one_hot, np.ndarray):
        return None
    if len(one_hot.shape) != 2:
        return None
    if one_hot.size == 0:
        return None

    # Get dimensions
    classes, m = one_hot.shape

    # Check if it's a valid one-hot matrix
    # Each column should have exactly one 1 and the rest 0s
    if not np.allclose(np.sum(one_hot, axis=0), 1):
        return None

    # Check if all values are 0 or 1
    if not np.all(np.logical_or(one_hot == 0, one_hot == 1)):
        return None

    # Find the index of the maximum value in each column
    # This gives us the class label for each example
    labels = np.argmax(one_hot, axis=0)

    return labels
