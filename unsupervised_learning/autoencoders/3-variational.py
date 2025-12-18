#!/usr/bin/env python3
"""
Module for creating a variational autoencoder.
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder model.

    Args:
        input_dims: integer containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each hidden
                       layer in the encoder
        latent_dims: integer containing the dimensions of the latent space
                     representation

    Returns:
        encoder: the encoder model with outputs (latent, mean, log_variance)
        decoder: the decoder model
        auto: the full variational autoencoder model
    """

    # ---------- ENCODER ----------
    inputs = keras.Input(shape=(input_dims,))
    x = inputs

    # Hidden layers with ReLU
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)

    # Mean and log variance layers (no activation)
    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    # Sampling layer (reparameterization trick)
    def sampling(args):
        mean, log_var = args
        epsilon = keras.backend.random_normal(shape=keras.backend.shape(mean))
        return mean + keras.backend.exp(0.5 * log_var) * epsilon

    z = keras.layers.Lambda(
        sampling,
        output_shape=(latent_dims,)
    )([z_mean, z_log_var])

    # encoder outputs: latent, mean, log_var
    encoder = keras.Model(inputs=inputs, outputs=[z, z_mean, z_log_var])

    # ---------- DECODER ----------
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs

    # Hidden layers in reverse order with ReLU
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)

    # Output layer with sigmoid activation
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(inputs=latent_inputs, outputs=outputs)

    # ---------- FULL AUTOENCODER ----------
    auto_inputs = keras.Input(shape=(input_dims,))
    encoded, mu, log_sigma = encoder(auto_inputs)
    decoded = decoder(encoded)

    auto = keras.Model(inputs=auto_inputs, outputs=decoded)
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    
    return encoder, decoder, auto
