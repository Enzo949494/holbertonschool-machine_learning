#!/usr/bin/env python3
"""
Module calculating the maximization step in EM algorithm GMM.

This module provides functionality to update GMM parameters
in the maximization step of the EM algorithm.
"""

import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step of the EM algorithm a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) - the dataset
        g: numpy.ndarray of shape (k, n) - posterior probabilities

    Returns:
        pi: numpy.ndarray of shape (k,) - updated priors each cluster
        m: numpy.ndarray of shape (k, d) - updated centroid means
        S: numpy.ndarray of shape (k, d, d) - updated covariance matrices
        or None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    if X.shape[0] != g.shape[1]:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    # Calculate N_k: sum of posterior probabilities each cluster
    N_k = np.sum(g, axis=1)

    # Avoid division by zero
    if np.any(N_k == 0):
        return None, None, None

    # Validate g: values should be between 0 and 1
    if np.any(g < 0) or np.any(g > 1):
        return None, None, None

    # Update priors
    pi = N_k / n

    # Update means
    m = (g @ X) / N_k[:, np.newaxis]

    # Update covariance matrices
    S = np.zeros((k, d, d))
    for i in range(k):
        X_centered = X - m[i]
        weighted_cov = (g[i] * X_centered.T) @ X_centered
        S[i] = weighted_cov / N_k[i]

    return pi, m, S