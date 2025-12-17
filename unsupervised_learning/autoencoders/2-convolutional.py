#!/usr/bin/env python3
"""
Module for creating a convolutional autoencoder.
"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder model.

    Args:
        input_dims: tuple of integers containing dimensions of the model input
        filters: list containing the number of filters for each convolutional
                 layer in the encoder
        latent_dims: tuple of integers containing the dimensions of the
                     latent space representation

    Returns:
        encoder: the encoder model
        decoder: the decoder model
        auto: the full convolutional autoencoder model
    """

    # Encoder
    inputs = keras.Input(shape=input_dims)
    x = inputs

    # Add convolutional layers with max pooling
    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), padding='same',
                                activation='relu')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    encoder = keras.Model(inputs=inputs, outputs=x)

    # Decoder
    latent_inputs = keras.Input(shape=latent_dims)
    x = latent_inputs

    # Reverse filters for the decoder
    rev_filters = list(reversed(filters))

    # All convs (same padding) + upsampling except the second to last conv
    for f in rev_filters[:-1]:
        x = keras.layers.Conv2D(f, (3, 3), padding='same',
                                activation='relu')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

    # Second to last convolution with valid padding + upsampling
    x = keras.layers.Conv2D(rev_filters[-1], (3, 3), padding='valid',
                            activation='relu')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)

    # Last convolution with sigmoid activation and input channels
    outputs = keras.layers.Conv2D(input_dims[-1], (3, 3), padding='same',
                                  activation='sigmoid')(x)

    decoder = keras.Model(inputs=latent_inputs, outputs=outputs)

    # Full convolutional autoencoder
    auto_inputs = keras.Input(shape=input_dims)
    encoded = encoder(auto_inputs)
    decoded = decoder(encoded)

    auto = keras.Model(inputs=auto_inputs, outputs=decoded)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
