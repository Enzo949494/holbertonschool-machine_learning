#!/usr/bin/env python3
"""
Module for Gaussian Mixture Model clustering using scikit-learn.
"""

import sklearn.mixture


def gmm(X, k):
    """
    Calculates a GMM from a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) - dataset
        k: number of clusters

    Returns:
        pi: numpy.ndarray of shape (k,) - cluster priors
        m: numpy.ndarray of shape (k, d) - centroid means
        S: numpy.ndarray of shape (k, d, d) - covariance matrices
        clss: numpy.ndarray of shape (n,) - cluster indices
        bic: float - BIC value for the model
    """
    if not isinstance(X, type(X)) or X.ndim != 2:
        return None, None, None, None, None

    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None

    if k > X.shape[0]:
        return None, None, None, None, None

    # Create and fit GMM model
    gm = sklearn.mixture.GaussianMixture(n_components=k)
    gm.fit(X)

    pi = gm.weights_
    m = gm.means_
    S = gm.covariances_
    clss = gm.predict(X)
    bic = gm.bic(X)

    return pi, m, S, clss, bic
