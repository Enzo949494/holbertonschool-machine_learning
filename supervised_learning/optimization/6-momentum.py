#!/usr/bin/env python3
"""
Module for creating momentum optimizer in TensorFlow
"""

import tensorflow as tf


def create_momentum_op(learning_rate, beta1):
    """
    Creates the training operation for a neural network in TensorFlow using
    the gradient descent with momentum optimization algorithm

    Args:
        learning_rate: the learning rate
        beta1: the momentum weight

    Returns:
        the momentum optimizer
    """
    return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=beta1)
