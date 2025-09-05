#!/usr/bin/env python3
"""Module for one-hot encoding"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix

    Args:
        labels: vector of labels to convert
        classes: number of classes (if None, inferred from max label + 1)

    Returns:
        The one-hot matrix
    """
    # Convert labels to one-hot encoding using Keras utility
    return K.utils.to_categorical(labels, num_classes=classes)
