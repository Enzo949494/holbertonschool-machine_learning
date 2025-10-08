#!/usr/bin/env python3
"""Identity block implementation for ResNet"""

from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in Deep Residual Learning
    for Image Recognition (2015)

    Args:
        A_prev: output the previous layer
        filters: tuple or list containing F11, F3, F12
            F11: number of filters in the first 1x1 convolution
            F3: number of filters in the 3x3 convolution
            F12: number of filters in the second 1x1 convolution

    Returns:
        activated output of the identity block
    """
    F11, F3, F12 = filters

    # He normal initializer with seed 0
    initializer = K.initializers.HeNormal(seed=0)

    # First 1x1 convolution
    conv1 = K.layers.Conv2D(
        F11, (1, 1), padding='same',
        kernel_initializer=initializer)(A_prev)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(bn1)

    # 3x3 convolution
    conv2 = K.layers.Conv2D(
        F3, (3, 3), padding='same',
        kernel_initializer=initializer)(act1)
    bn2 = K.layers.BatchNormalization(axis=3)(conv2)
    act2 = K.layers.Activation('relu')(bn2)

    # 2nd 1x1 convolution
    conv3 = K.layers.Conv2D(
        F12, (1, 1), padding='same',
        kernel_initializer=initializer)(act2)
    bn3 = K.layers.BatchNormalization(axis=3)(conv3)

    # Add shortcut connection
    add = K.layers.Add()([bn3, A_prev])

    # Final activation
    output = K.layers.Activation('relu')(add)

    return output
