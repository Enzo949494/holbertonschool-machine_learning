#!/usr/bin/env python3
"""L2 Regularization Gradient Descent"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient descent
    with L2 regularization
    
    Args:
        Y: one-hot numpy.ndarray of shape (classes, m) with correct labels
        weights: dictionary of weights and biases of the neural network
        cache: dictionary of outputs of each layer of the neural network
        alpha: learning rate
        lambtha: L2 regularization parameter
        L: number of layers of the network
    """
    m = Y.shape[1]
    
    # Backward propagation
    # Start with output layer (softmax)
    dZ = cache[f'A{L}'] - Y
    
    for layer in range(L, 0, -1):
        # Current layer activations
        A_prev = cache[f'A{layer-1}']
        
        # Gradients for weights and biases
        dW = (1/m) * np.matmul(dZ, A_prev.T) + (lambtha/m) * weights[f'W{layer}']
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        # Update weights and biases
        weights[f'W{layer}'] -= alpha * dW
        weights[f'b{layer}'] -= alpha * db
        
        # Calculate dZ for previous layer (if not the first layer)
        if layer > 1:
            # dA for previous layer
            dA_prev = np.matmul(weights[f'W{layer}'].T, dZ)
            # dZ for previous layer (tanh derivative)
            dZ = dA_prev * (1 - cache[f'A{layer-1}']**2)