#!/usr/bin/env python3
"""
Module for initializing Gaussian Mixture Model parameters.

This module provides functionality to initialize the parameters
for a Gaussian Mixture Model (GMM) clustering algorithm.
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model.

    Args:
        X: numpy.ndarray of shape (n, d) - the dataset
        k: positive integer - number of clusters

    Returns:
        pi: numpy.ndarray of shape (k,) - priors for each cluster
        m: numpy.ndarray of shape (k, d) - centroid means for each cluster
        S: numpy.ndarray of shape (k, d, d) - covariance matrices
        or None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    if not isinstance(k, int) or k <= 0:
        return None, None, None

    n, d = X.shape

    if k > n:
        return None, None, None

    # Initialize priors evenly
    pi = np.full(k, 1 / k)

    # Initialize means using K-means
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None

    # Initialize covariance matrices as identity matrices
    S = np.identity(d)
    S = np.tile(S, (k, 1, 1))

    return pi, m, S
