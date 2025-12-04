#!/usr/bin/env python3
"""
Module for agglomerative hierarchical clustering.
"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) - dataset
        dist: maximum cophenetic distance for all clusters

    Returns:
        clss: numpy.ndarray of shape (n,) - cluster indices for each point
    """
    if not isinstance(X, type(X)) or X.ndim != 2:
        return None

    if not isinstance(dist, (int, float)) or dist <= 0:
        return None

    # Perform hierarchical clustering using Ward linkage
    Z = scipy.cluster.hierarchy.linkage(X, method='ward')

    # Cut the dendrogram at the specified distance
    clss = scipy.cluster.hierarchy.fcluster(Z, dist, criterion='distance')

    # Adjust cluster indices to start from 0
    clss = clss - 1

    # Plot dendrogram with colors
    plt.figure()
    scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist, no_labels=True)
    plt.show()

    return clss
