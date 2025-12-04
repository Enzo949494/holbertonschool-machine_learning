#!/usr/bin/env python3
"""
Module finding the best number of clusters using BIC.

This module provides functionality to find the optimal number of clusters
a Gaussian Mixture Model using the Bayesian Information Criterion.
"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters a GMM using BIC.

    Args:
        X: numpy.ndarray of shape (n, d) - the dataset
        kmin: positive integer - minimum number of clusters (default 1)
        kmax: positive integer - maximum number of clusters
        iterations: positive integer - max iterations EM (default 1000)
        tol: non-negative float - tolerance EM (default 1e-5)
        verbose: boolean - print EM info (default False)

    Returns:
        best_k: best value k based on BIC
        best_result: tuple (pi, m, S) best k
        l: numpy.ndarray of log likelihoods each k
        b: numpy.ndarray of BIC values each k
        or None, None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None

    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None

    if kmax is None:
        kmax = X.shape[0]

    if not isinstance(kmax, int) or kmax <= 0 or kmax < kmin:
        return None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None

    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    l = np.zeros(kmax - kmin + 1)
    b = np.zeros(kmax - kmin + 1)
    best_bic = np.inf
    best_k = None
    best_result = None

    # Loop through each k value
    for i, k in enumerate(range(kmin, kmax + 1)):
        pi, m, S, g, l_val = expectation_maximization(X, k, iterations, tol, verbose)
        
        if pi is None or m is None or S is None or g is None or l_val is None:
            return None, None, None, None

        l[i] = l_val

        # Calculate number of parameters
        # pi: k parameters (but sum to 1, so k-1 independent)
        # m: k * d parameters
        # S: k * d * (d+1) / 2 parameters (symmetric covariance matrices)
        p = (k - 1) + k * d + k * d * (d + 1) // 2

        # Calculate BIC
        b[i] = p * np.log(n) - 2 * l_val

        # Track best BIC (lowest is best)
        if b[i] < best_bic:
            best_bic = b[i]
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, l, b