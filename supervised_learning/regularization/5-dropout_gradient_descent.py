#!/usr/bin/env python3
"""Dropout Gradient Descent"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent.

    Args:
        Y: one-hot numpy.ndarray, shape (classes, m), correct labels
        weights: dictionary of the weights and biases
        cache: dictionary of activation outputs and dropout masks
        alpha: learning rate
        keep_prob: probability of keeping a neuron active during dropout
        L: number of layers in the network

    Returns:
        None. Updates weights in place.
    """
    m = Y.shape[1]
    dZ = cache['A{}'.format(L)] - Y  # Gradient for output layer (softmax)

    for layer in range(L, 0, -1):
        A_prev = cache['A{}'.format(layer - 1)]
        W = weights['W{}'.format(layer)]

        # Compute gradients of weights and biases
        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if layer > 1:
            # Backpropagate the gradient through layer (tanh + dropout)
            dA_prev = np.matmul(W.T, dZ)
            D = cache['D{}'.format(layer - 1)]
            dA_prev = dA_prev * D              # Apply dropout mask
            dA_prev /= keep_prob               # Scale gradient to keep expect
            A_prev = cache['A{}'.format(layer - 1)]
            dZ = dA_prev * (1 - A_prev ** 2)  # Derivative of tanh activation

        weights['W{}'.format(layer)] -= alpha * dW
        weights['b{}'.format(layer)] -= alpha * db
