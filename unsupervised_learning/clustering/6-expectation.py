#!/usr/bin/env python3
"""
Module for calculating the expectation step in EM algorithm for GMM.

This module provides functionality to calculate posterior probabilities
and log likelihood in the expectation step of the EM algorithm.
"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step of the EM algorithm for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) - the dataset
        pi: numpy.ndarray of shape (k,) - priors for each cluster
        m: numpy.ndarray of shape (k, d) - centroid means for each cluster
        S: numpy.ndarray of shape (k, d, d) - covariance matrices

    Returns:
        g: numpy.ndarray of shape (k, n) - posterior probabilities
        l: total log likelihood
        or None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None

    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None

    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None

    if X.shape[1] != m.shape[1] or X.shape[1] != S.shape[1]:
        return None, None

    if pi.shape[0] != m.shape[0] or pi.shape[0] != S.shape[0]:
        return None, None

    k, d = m.shape
    n = X.shape[0]

    # Initialize g matrix (k, n)
    g = np.zeros((k, n))

    # Calculate PDF for each cluster
    for i in range(k):
        P = pdf(X, m[i], S[i])
        if P is None:
            return None, None
        g[i] = pi[i] * P

    # Normalize to get posterior probabilities
    g_sum = np.sum(g, axis=0)
    
    # Avoid division by zero
    g_sum = np.maximum(g_sum, 1e-300)
    g = g / g_sum

    # Calculate total log likelihood
    l = np.sum(np.log(g_sum))

    return g, l