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
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
        not isinstance(k, int) or k <= 0 or
        not isinstance(iterations, int) or iterations <= 0 or
        not isinstance(tol, (int, float)) or tol < 0 or
        not isinstance(verbose, bool)):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None:
        return None, None, None, None, None

    l_prev = 0

    for i in range(iterations):
        # E-step
        g, l = expectation(X, pi, m, S)
        if g is None:
            return None, None, None, None, None

        # verbose
        if verbose and (i % 10 == 0):
            print("Log Likelihood after {} iterations: {}".format(
                i, round(l, 5)))

        # critère d'arrêt
        if i > 0 and abs(l - l_prev) <= tol:
            break

        # M-step
        pi, m, S = maximization(X, g)
        if pi is None:
            return None, None, None, None, None

        l_prev = l

    # E-step finale avec les derniers paramètres
    g, l = expectation(X, pi, m, S)
    if g is None:
        return None, None, None, None, None

    # dernier print si besoin
    if verbose and (i % 10 != 0):
        print("Log Likelihood after {} iterations: {}".format(
            i, round(l, 5)))

    return pi, m, S, g, l
