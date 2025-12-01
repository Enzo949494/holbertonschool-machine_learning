#!/usr/bin/env python3
"""
Module for K-means clustering initialization.

This module provides functionality to initialize cluster centroids
for K-means clustering algorithm using uniform random distribution.
"""

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids K-means clustering.

    Args:
        X: numpy.ndarray of shape (n, d) - the dataset
        k: positive integer - number of clusters

    Returns:
        numpy.ndarray of shape (k, d) with initialized centroids,
        or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape

    if k > n:
        return None

    # Get min and max values each dimension
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    # Initialize centroids with multivariate uniform distribution
    centroids = np.random.uniform(min_vals, max_vals, size=(k, d))

    return centroids
