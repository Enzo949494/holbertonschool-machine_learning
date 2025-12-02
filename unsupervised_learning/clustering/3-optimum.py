#!/usr/bin/env python3
"""
Module for finding the optimum number of clusters.

This module provides functionality to test different cluster sizes
and find the optimum k using the elbow method via variance analysis.
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance.

    Args:
        X: numpy.ndarray of shape (n, d) - the dataset
        kmin: positive integer - minimum number of clusters (default 1)
        kmax: positive integer - maximum number of clusters
        iterations: positive integer - max iterations K-means (default 1000)

    Returns:
        results: list of K-means outputs each cluster size
        d_vars: list of variance differences from smallest cluster size
        or None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    if not isinstance(kmin, int) or kmin <= 0:
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    if not isinstance(kmax, int) or kmax <= 0 or kmax < kmin:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    if kmax - kmin < 1:
        return None, None

    results = []
    variances = []

    # Loop through each k value
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None

        results.append((C, clss))
        var = variance(X, C)
        if var is None:
            return None, None

        variances.append(var)

    # Calculate differences from the first variance
    d_vars = [variances[0] - v for v in variances]

    return results, d_vars
