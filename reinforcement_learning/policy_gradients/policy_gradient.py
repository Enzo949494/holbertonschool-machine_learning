#!/usr/bin/env python3
import numpy as np


def policy(matrix, weight):
    """
    Computes the policy (action probabilities) via softmax.
    matrix: state, shape (1, state_size)
    weight: weight matrix, shape (state_size, action_size)
    returns: action probability distribution, shape (1, action_size)
    """
    z = matrix @ weight          # linear scores: (1, action_size)
    z -= np.max(z, axis=1, keepdims=True)  # numerical stability
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
