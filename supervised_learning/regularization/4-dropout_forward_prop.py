#!/usr/bin/env python3
"""Forward propagation with Dropout"""

import numpy as np

def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout

    Args:
        X: numpy.ndarray (nx, m) input data
        weights: dict of weights and biases
        L: number of layers
        keep_prob: probability to keep a node

    Returns:
        dict with outputs and dropout masks
    """
    cache = {}
    cache['A0'] = X

    for l in range(1, L + 1):
        W = weights['W{}'.format(l)]
        b = weights['b{}'.format(l)]
        A_prev = cache['A{}'.format(l - 1)]
        Z = np.matmul(W, A_prev) + b

        if l == L:
            # Output layer: softmax
            t = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = t / np.sum(t, axis=0, keepdims=True)
            cache['A{}'.format(l)] = A
        else:
            # Hidden layers: tanh + dropout
            A = np.tanh(Z)
            D = np.random.binomial(1, keep_prob, size=A.shape)
            A = (A * D) / keep_prob
            cache['A{}'.format(l)] = A
            cache['D{}'.format(l)] = D

    return cache