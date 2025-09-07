#!/usr/bin/env python3
"""Module for saving and loading Keras models"""

import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model

    Args:
        network: the model to save
        filename: the path of the file that the model should be saved to

    Returns:
        None
    """
    # Save the entire model (architecture + weights + optimizer state)
    network.save(filename)


def load_model(filename):
    """
    Loads an entire model

    Args:
        filename: the path of the file that the model should be loaded from

    Returns:
        The loaded model
    """
    # Load the entire model from file
    return K.models.load_model(filename)
