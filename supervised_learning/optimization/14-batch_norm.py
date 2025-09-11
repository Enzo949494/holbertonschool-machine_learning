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
        activation: activation function that should be used on the output of the layer
    
    Returns:
        a tensor of the activated output for the layer
    """
    # Dense layer
    Z = tf.keras.layers.Dense(
        n, 
        kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg'),
        use_bias=False
    )(prev)
    
    # Batch normalization
    Z_norm = tf.keras.layers.BatchNormalization(
        beta_initializer='zeros',
        gamma_initializer='ones',
        epsilon=1e-7
    )(Z)
    
    # Activation
    return activation(Z_norm)