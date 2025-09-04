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
    model = K.Sequential()
    
    # Add layers sequentially
    for i in range(len(layers)):
        if i == 0:
            # First layer needs input_shape
            model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha),
                input_shape=(nx,)
            ))
        else:
            # Hidden and output layers
            model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha)
            ))
        
        # Add dropout after each layer except the last one
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    
    return model