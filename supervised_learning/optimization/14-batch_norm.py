#!/usr/bin/env python3
"""
Module for creating batch normalization layer in TensorFlow
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow

    Args:
        prev: activated output of the previous layer
        n: number of nodes in the layer to be created
        activation: activation function that should be used
                    on the output of the layer

    Returns:
        a tensor of the activated output for the layer
    """
    # Dense layer
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(
        units=n,
        use_bias=False,
        kernel_initializer=initializer
    )(prev)

    # Calculate mean and variance for batch normalization
    mean, variance = tf.nn.moments(dense, axes=[0])

    # Create trainable parameters gamma and beta
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)

    # Apply batch normalization
    batch_norm = tf.nn.batch_normalization(
        x=dense,
        mean=mean,
        variance=variance,
        offset=beta,      # beta parameter
        scale=gamma,      # gamma parameter
        variance_epsilon=1e-7
    )

    # Apply activation
    return activation(batch_norm)
