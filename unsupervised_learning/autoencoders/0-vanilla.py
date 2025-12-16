#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras

def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder model.
    
    Args:
        input_dims: integer containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each hidden layer in the encoder
        latent_dims: integer containing the dimensions of the latent space representation
    
    Returns:
        encoder: the encoder model
        decoder: the decoder model
        auto: the full autoencoder model
    """
    
    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    
    # Add hidden layers with relu activation
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)
    
    # Add latent layer
    latent = keras.layers.Dense(latent_dims, activation='relu')(x)
    
    encoder = keras.Model(inputs=inputs, outputs=latent)
    
    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    
    # Add hidden layers in reverse order with relu activation
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)
    
    # Add output layer with sigmoid activation
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    
    decoder = keras.Model(inputs=latent_inputs, outputs=outputs)
    
    # Full autoencoder
    auto_inputs = keras.Input(shape=(input_dims,))
    encoded = encoder(auto_inputs)
    decoded = decoder(encoded)
    
    auto = keras.Model(inputs=auto_inputs, outputs=decoded)
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    
    return encoder, decoder, auto
