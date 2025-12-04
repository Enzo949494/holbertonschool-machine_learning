#!/usr/bin/env python3
"""
Module for K-means clustering using scikit-learn.
"""

import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means clustering on a dataset.
    
    Args:
        X: numpy.ndarray of shape (n, d) - dataset
        k: number of clusters
    
    Returns:
        C: numpy.ndarray of shape (k, d) - centroid means
        clss: numpy.ndarray of shape (n,) - cluster index for each point
    """
    if not isinstance(X, type(X)) or X.ndim != 2:
        return None, None

    if not isinstance(k, int) or k <= 0:
        return None, None

    if k > X.shape[0]:
        return None, None

    # Create and fit KMeans model
    km = sklearn.cluster.KMeans(n_clusters=k, random_state=None)
    clss = km.fit_predict(X)
    C = km.cluster_centers_

    return C, clss
