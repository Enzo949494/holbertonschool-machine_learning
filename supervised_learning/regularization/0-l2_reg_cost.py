#!/usr/bin/env python3
"""
Module for calculating L2 regularization cost
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization
    
    Args:
        cost: cost of the network without L2 regularization
        lambtha: regularization parameter
        weights: dictionary of the weights and biases (numpy.ndarrays) 
                 of the neural network
        L: number of layers in the neural network
        m: number of data points used
    
    Returns:
        the cost of the network accounting for L2 regularization
    """
    # Initialize L2 regularization term
    l2_reg_term = 0
    
    # Sum the squared weights for all layers
    for layer in range(1, L + 1):
        weight_key = f'W{layer}'
        if weight_key in weights:
            # Add squared Frobenius norm of weights
            l2_reg_term += np.sum(np.square(weights[weight_key]))
    
    # Calculate L2 regularized cost
    # Cost = Original Cost + (lambda / (2 * m)) * sum(W^2)
    l2_cost = cost + (lambtha / (2 * m)) * l2_reg_term
    
    return l2_cost