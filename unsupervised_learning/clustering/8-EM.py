#!/usr/bin/env python3
"""
Module Expectation-Maximization algorithm Gaussian Mixture Model.

This module implements the complete EM algorithm GMM clustering
with convergence checking and optional verbose output.
"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs expectation maximization a Gaussian Mixture Model.

    Args:
        X: numpy.ndarray of shape (n, d) - the dataset
        k: positive integer - number of clusters
        iterations: positive integer - max iterations (default 1000)
        tol: non-negative float - tolerance early stopping (default 1e-5)
        verbose: boolean - print log likelihood info (default False)

    Returns:
        pi: numpy.ndarray of shape (k,) - priors each cluster
        m: numpy.ndarray of shape (k, d) - centroid means each cluster
        S: numpy.ndarray of shape (k, d, d) - covariance matrices
        g: numpy.ndarray of shape (k, n) - posterior probabilities
        l: total log likelihood
        or None, None, None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None

    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None

    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None, None

    # Initialize parameters
    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    # EM iterations
    for i in range(iterations):
        # Expectation step
        g, l = expectation(X, pi, m, S)
        if g is None or l is None:
            return None, None, None, None, None

        # Print log likelihood if verbose
        if verbose and (i % 10 == 0 or i == iterations - 1):
            print("Log Likelihood after {} iterations: {}".format(
                i, round(l, 5)))

        # Maximization step
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        # Check convergence after maximization
        g, l_new = expectation(X, pi, m, S)
        if g is None or l_new is None:
            return None, None, None, None, None

        # Print after convergence if verbose
        if verbose and abs(l_new - l) <= tol:
            print("Log Likelihood after {} iterations: {}".format(
                i + 1, round(l_new, 5)))

        if abs(l_new - l) <= tol:
            l = l_new
            break

        l = l_new

    # Final expectation step to get final g and l
    g, l = expectation(X, pi, m, S)

    return pi, m, S, g, l