#!/usr/bin/env python3
"""Dropout Gradient Descent"""

import numpy as np

def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent.
    """
    m = Y.shape[1]
    dZ = cache['A{}'.format(L)] - Y

    for layer in range(L, 0, -1):
        A_prev = cache['A{}'.format(layer - 1)]
        W = weights['W{}'.format(layer)]
        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Update weights and biases
        weights['W{}'.format(layer)] -= alpha * dW
        weights['b{}'.format(layer)] -= alpha * db

        if layer > 1:
            # Backpropagate through tanh and dropout
            dA_prev = np.matmul(W.T, dZ)
            # Appliquer le masque Dropout du forward
            D = cache['D{}'.format(layer - 1)]
            dA_prev = dA_prev * D
            dA_prev = dA_prev / keep_prob
            # Dérivée de tanh
            A_prev = cache['A{}'.format(layer - 1)]
            dZ = dA_prev * (1 - A_prev ** 2)