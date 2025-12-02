#!/usr/bin/env python3
"""
Module for K-means clustering algorithm.

This module implements the K-means clustering algorithm to partition
a dataset into k clusters based on feature similarity.
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) - the dataset
        k: positive integer - number of clusters
        iterations: positive integer - maximum number of iterations (default 1000)

    Returns:
        C: numpy.ndarray of shape (k, d) - centroid means each cluster
        clss: numpy.ndarray of shape (n,) - cluster index each data point
        or None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    if not isinstance(k, int) or k <= 0:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    if k > n:
        return None, None

    # Get min and max values initialization
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    # Initialize centroids with multivariate uniform distribution
    C = np.random.uniform(min_vals, max_vals, size=(k, d))

    # K-means iterations
    for _ in range(iterations):
        # Store previous centroids to check convergence
        C_prev = C.copy()

        # Assignment step: assign each point to nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        # Update step: recalculate centroids
        C = np.zeros((k, d))
        for i in range(k):
            cluster_points = X[clss == i]
            if cluster_points.shape[0] == 0:
                # Reinitialize centroid if cluster is empty
                C[i] = np.random.uniform(min_vals, max_vals, size=d)
            else:
                C[i] = cluster_points.mean(axis=0)

        # Check convergence
        if np.allclose(C, C_prev):
            break

    return C, clss