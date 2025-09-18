#!/usr/bin/env python3
"""Creates a layer with Dropout regularization in TensorFlow"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout

    Args:
        prev: tensor containing the output of the previous layer
        n: number of nodes the new layer should contain
        activation: activation function for the new layer
        keep_prob: probability that a node will be kept
        training: boolean, True for training mode (dropout active)

    Returns:
        the output of the new layer
    """
    dense = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_avg'
        )
    )(prev)
    dropout = tf.keras.layers.Dropout(rate=1 - keep_prob)
    return dropout(dense, training=training)
