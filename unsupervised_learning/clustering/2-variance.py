#!/usr/bin/env python3
"""
Module calculating intra-cluster variance.

This module provides functionality to calculate the total intra-cluster
variance for a dataset given cluster centroids.
"""

import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance a data set.

    Args:
        X: numpy.ndarray of shape (n, d) - the dataset
        C: numpy.ndarray of shape (k, d) - centroid means each cluster

    Returns:
        var: total variance, or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None

    if X.shape[1] != C.shape[1]:
        return None

    # Calculate distances from each point to all centroids
    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)

    # Find minimum distance each point (distance to nearest centroid)
    min_distances = np.min(distances, axis=1)

    # Calculate variance: sum of squared distances
    var = np.sum(min_distances ** 2)

    return var
