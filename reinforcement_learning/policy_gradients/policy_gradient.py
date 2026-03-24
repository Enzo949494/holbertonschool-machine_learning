#!/usr/bin/env python3
"""
Policy Gradient Module

This module contains functions for computing policy gradients
in reinforcement learning algorithms.
"""
import numpy as np


def policy(matrix, weight):
    """
    Computes the policy (action probabilities) using softmax.

    Parameters
    ----------
    matrix : np.ndarray
        State vector with shape (1, state_size)
    weight : np.ndarray
        Weight matrix with shape (state_size, action_size)

    Returns
    -------
    np.ndarray
        Action probability distribution with shape (1, action_size)
    """
    z = matrix @ weight          # linear scores: (1, action_size)
    z -= np.max(z, axis=1, keepdims=True)  # numerical stability
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
