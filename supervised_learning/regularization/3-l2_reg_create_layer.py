#!/usr/bin/env python3
"""L2 Regularization Layer Creation with TensorFlow"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer in TensorFlow that includes L2 regularization
    """
    regularizer = tf.keras.regularizers.l2(lambtha) if lambtha > 0 else None
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=regularizer,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_avg'
        )
    )
    return layer(prev)