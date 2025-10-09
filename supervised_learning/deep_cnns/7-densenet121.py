#!/usr/bin/env python3
"""DenseNet-121 architecture implementation"""

from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described in Densely Connected
    Convolutional Networks

    Args:
        growth_rate: the growth rate
        compression: the compression factor

    Returns:
        the keras model
    """
    # He normal initializer with seed 0
    initializer = K.initializers.HeNormal(seed=0)

    # Input layer
    input_layer = K.Input(shape=(224, 224, 3))

    # Initial convolution: BN -> ReLU -> Conv 7x7, stride 2
    bn = K.layers.BatchNormalization(axis=3)(input_layer)
    act = K.layers.Activation('relu')(bn)
    conv1 = K.layers.Conv2D(
        64, (7, 7), strides=(2, 2), padding='same',
        kernel_initializer=initializer)(act)

    # Max pooling 3x3, stride 2
    pool1 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same')(conv1)

    # Initialize number of filters
    nb_filters = 64

    # Dense Block 1 (6 layers)
    dense1, nb_filters = dense_block(pool1, nb_filters, growth_rate, 6)

    # Transition Layer 1
    trans1, nb_filters = transition_layer(dense1, nb_filters, compression)

    # Dense Block 2 (12 layers)
    dense2, nb_filters = dense_block(trans1, nb_filters, growth_rate, 12)

    # Transition Layer 2
    trans2, nb_filters = transition_layer(dense2, nb_filters, compression)

    # Dense Block 3 (24 layers)
    dense3, nb_filters = dense_block(trans2, nb_filters, growth_rate, 24)

    # Transition Layer 3
    trans3, nb_filters = transition_layer(dense3, nb_filters, compression)

    # Dense Block 4 (16 layers) - no transition after
    dense4, nb_filters = dense_block(trans3, nb_filters, growth_rate, 16)

    # Global Average Pooling 7x7
    avg_pool = K.layers.AveragePooling2D((7, 7), strides=(1, 1))(dense4)

    # Fully connected layer with 1000 outputs (softmax)
    output = K.layers.Dense(
        1000, activation='softmax',
        kernel_initializer=initializer)(avg_pool)

    # Create model
    model = K.Model(inputs=input_layer, outputs=output)

    return model
