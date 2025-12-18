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

    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs

    # Add hidden layers with relu activation
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)

    # Mean and log variance layers (no activation)
    mean = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    # Sampling layer
    def sampling(args):
        mean, log_var = args
        epsilon = keras.backend.random_normal(shape=(keras.backend.shape(mean)[0], latent_dims))
        return mean + keras.backend.exp(log_var / 2) * epsilon

    latent = keras.layers.Lambda(sampling)([mean, log_var])

    encoder = keras.Model(inputs=inputs, outputs=[latent, mean, log_var])

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs

    # Add hidden layers in reverse order with relu activation
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)

    # Output layer with sigmoid activation
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(inputs=latent_inputs, outputs=outputs)

    # Full variational autoencoder
    auto_inputs = keras.Input(shape=(input_dims,))
    encoded, mu, log_sigma = encoder(auto_inputs)
    decoded = decoder(encoded)

    # Custom layer for VAE loss
    class VAELoss(keras.layers.Layer):
        def call(self, inputs):
            y_true, y_pred, mu, log_sigma = inputs
            reconstruction_loss = keras.losses.binary_crossentropy(y_true, y_pred)
            reconstruction_loss *= input_dims
            kl_loss = 1 + log_sigma - keras.ops.square(mu) - keras.ops.exp(log_sigma)
            kl_loss = keras.ops.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            return reconstruction_loss + kl_loss

    loss_layer = VAELoss()
    loss_output = loss_layer([auto_inputs, decoded, mu, log_sigma])

    auto = keras.Model(inputs=auto_inputs, outputs=loss_output)
    auto.compile(optimizer='adam', loss=lambda y_true, y_pred: y_pred)

    return encoder, decoder, auto