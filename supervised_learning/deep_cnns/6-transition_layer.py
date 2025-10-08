#!/usr/bin/env python3
"""Transition layer implementation for DenseNet"""

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in Densely Connected
    Convolutional Networks

    Args:
        X: output from the previous layer
        nb_filters: integer representing the number of filters in X
        compression: compression factor for the transition layer

    Returns:
        The output of the transition layer and the number of filters
        within the output
    """
    # He normal initializer with seed 0
    initializer = K.initializers.HeNormal(seed=0)

    # Calculate number of filters after compression
    nb_filters = int(nb_filters * compression)

    # Batch Normalization
    bn = K.layers.BatchNormalization(axis=3)(X)

    # ReLU activation
    act = K.layers.Activation('relu')(bn)

    # 1x1 Convolution for compression
    conv = K.layers.Conv2D(
        nb_filters, (1, 1), padding='same',
        kernel_initializer=initializer)(act)

    # Average Pooling 2x2 with stride 2
    pool = K.layers.AveragePooling2D((2, 2), strides=(2, 2))(conv)

    return pool, nb_filters
