#!/usr/bin/env python3
"""Module for building neural networks with Keras"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library

    Args:
        nx: number of input features to the network
        layers: list containing the number of nodes in each layer
        activations: list containing activation functions for each layer
        lambtha: L2 regularization parameter
        keep_prob: probability that a node will be kept for dropout

    Returns:
        The keras model
    """
    # Create an empty Sequential model (layers stacked one after another)
    model = K.Sequential()
    
    # Loop through each layer to build the network architecture
    for i in range(len(layers)):
        # Check if this is the first layer (needs input shape)
        if i == 0:
            # First layer: specify input_shape to define network input
            model.add(K.layers.Dense(
                layers[i],                                  # Number of neurons in this layer
                activation=activations[i],                  # Activation function (tanh, relu, softmax, etc.)
                kernel_regularizer=K.regularizers.l2(lambtha),  # L2 regularization to prevent overfitting
                input_shape=(nx,)                           # Input shape (only needed for first layer)
            ))
        else:
            # Hidden and output layers: Keras infers input shape automatically
            model.add(K.layers.Dense(
                layers[i],                                  # Number of neurons in this layer
                activation=activations[i],                  # Activation function for this layer
                kernel_regularizer=K.regularizers.l2(lambtha)  # L2 regularization on weights
            ))
        
        # Add dropout layer between hidden layers (not after output layer)
        if i < len(layers) - 1:
            # Dropout for regularization: randomly drop neurons during training
            # Convert keep_prob to drop_rate: drop_rate = 1 - keep_prob
            model.add(K.layers.Dropout(1 - keep_prob))
    
    # Return the constructed model (still needs compilation before training)
    return model
