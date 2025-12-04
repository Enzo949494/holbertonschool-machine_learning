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

    l_prev = 0
    g = None
    l = None

    # EM iterations
    for i in range(iterations + 1):
        # E-step: compute g and current log-likelihood
        g, l = expectation(X, pi, m, S)
        if g is None or l is None:
            return None, None, None, None, None

        # Print log likelihood every 10 iterations (including 0)
        if verbose and (i % 10 == 0):
            print("Log Likelihood after {} iterations: {}".format(
                i, round(l, 5)))

        # Check convergence (after computing l)
        if i > 0 and abs(l - l_prev) <= tol:
            break

        # M-step: update parameters
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        l_prev = l

    # After loop, if last iteration is not multiple of 10, print once more
    if verbose and (i % 10 != 0):
        print("Log Likelihood after {} iterations: {}".format(
            i, round(l, 5)))

    g, l = expectation(X, pi, m, S)
    if g is None or l is None:
        return None, None, None, None, None
    
    # g et l correspondent déjà à la dernière E-step effectuée
    return pi, m, S, g, l
