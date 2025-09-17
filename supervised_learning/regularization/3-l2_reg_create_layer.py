#!/usr/bin/env python3


import tensorflow as tf

def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a dense neural network layer with L2 regularization and
    VarianceScaling weight initialization.

    Args:
        prev: tensor output from previous layer
        n: number of nodes for the new layer
        activation: activation function to apply
        lambtha: L2 regularization parameter

    Returns:
        tensor output of the new layer with L2 regularization
    """
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.L2(l2=lambtha),
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')
    )(prev)

    return layer
