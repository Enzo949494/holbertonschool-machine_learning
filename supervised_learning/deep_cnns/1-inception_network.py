#!/usr/bin/env python3
"""Inception Network implementation"""

from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds the inception network as described in Going Deeper with Convolutions

    Returns:
        the keras model
    """
    # Input layer
    input_layer = K.Input(shape=(224, 224, 3))

    # Initial convolution layers
    conv1 = K.layers.Conv2D(
        64, (7, 7), strides=(2, 2), padding='same',
        activation='relu')(input_layer)
    max_pool1 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same')(conv1)

    # Convolution before inception blocks
    conv2_reduce = K.layers.Conv2D(
        64, (1, 1), padding='same', activation='relu')(max_pool1)
    conv2 = K.layers.Conv2D(
        192, (3, 3), padding='same', activation='relu')(conv2_reduce)
    max_pool2 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same')(conv2)

    # Inception 3a
    inception_3a = inception_block(max_pool2, [64, 96, 128, 16, 32, 32])

    # Inception 3b
    inception_3b = inception_block(inception_3a, [128, 128, 192, 32, 96, 64])
    max_pool3 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same')(inception_3b)

    # Inception 4a
    inception_4a = inception_block(max_pool3, [192, 96, 208, 16, 48, 64])

    # Inception 4b
    inception_4b = inception_block(inception_4a, [160, 112, 224, 24, 64, 64])

    # Inception 4c
    inception_4c = inception_block(inception_4b, [128, 128, 256, 24, 64, 64])

    # Inception 4d
    inception_4d = inception_block(inception_4c, [112, 144, 288, 32, 64, 64])

    # Inception 4e
    inception_4e = inception_block(inception_4d, [256, 160, 320, 32, 128, 128])
    max_pool4 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same')(inception_4e)

    # Inception 5a
    inception_5a = inception_block(max_pool4, [256, 160, 320, 32, 128, 128])

    # Inception 5b
    inception_5b = inception_block(inception_5a, [384, 192, 384, 48, 128, 128])

    # Average pooling
    avg_pool = K.layers.AveragePooling2D((7, 7), strides=(1, 1))(inception_5b)

    # Dropout
    dropout = K.layers.Dropout(0.4)(avg_pool)
    # Fully connected layer (Dense with softmax)
    output = K.layers.Dense(1000, activation='softmax')(dropout)

    # Create model
    model = K.Model(inputs=input_layer, outputs=output)

    return model
