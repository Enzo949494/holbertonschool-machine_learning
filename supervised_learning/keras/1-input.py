#!/usr/bin/env python3
"""Module for building neural networks with Keras using functional API"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library using functional API

    Args:
        nx: number of input features to the network
        layers: list containing the number of nodes in each layer
        activations: list containing activation functions for each layer
        lambtha: L2 regularization parameter
        keep_prob: probability that a node will be kept for dropout

    Returns:
        The keras model
    """
    # Create the input layer (defines the shape of input data)
    inputs = K.Input(shape=(nx,))
    
    # Start with the input layer as the current layer
    x = inputs
    
    # Build layers sequentially using functional API
    for i in range(len(layers)):
        # Add Dense layer with specified neurons, activation, and L2 regularization
        x = K.layers.Dense(
            layers[i],                                  # Number of neurons in this layer
            activation=activations[i],                  # Activation function for this layer
            kernel_regularizer=K.regularizers.l2(lambtha)  # L2 regularization on weights
        )(x)  # Apply layer to previous layer output
        
        # Add dropout layer between hidden layers (not after output layer)
        if i < len(layers) - 1:
            # Dropout for regularization: randomly drop neurons during training
            # Convert keep_prob to drop_rate: drop_rate = 1 - keep_prob
            x = K.layers.Dropout(1 - keep_prob)(x)
    
    # Create the model by specifying inputs and outputs
    model = K.Model(inputs=inputs, outputs=x)
    
    # Return the constructed model
    return model