#!/usr/bin/env python3
"""
Expectation-Maximization for a Gaussian Mixture Model
"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization for a GMM

    X: numpy.ndarray of shape (n, d) containing the data set
    k: positive integer, number of clusters
    iterations: maximum number of iterations
    tol: tolerance on the change in log likelihood for early stopping
    verbose: if True, prints log likelihood information

    Returns: pi, m, S, g, l or (None, None, None, None, None) on failure
    """
    # basic checks
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
        not isinstance(k, int) or k <= 0 or
        not isinstance(iterations, int) or iterations <= 0 or
        not isinstance(tol, (int, float)) or tol < 0 or
        not isinstance(verbose, bool)):
        return None, None, None, None, None

    # initialization
    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    l_prev = 0.0

    for i in range(iterations + 1):

        # E-step: compute posterior probabilities and current log likelihood
        g, l = expectation(X, pi, m, S)
        if g is None or l is None:
            return None, None, None, None, None

        # print every 10 iterations including 0
        if verbose and (i % 10 == 0):
            print("Log Likelihood after {} iterations: {}".format(
                i, round(l, 5)))

        # convergence test (skip i = 0)
        if i > 0 and abs(l - l_prev) <= tol:
            # on s'arrête ici : i est l'indice de la dernière itération
            break

        if i == iterations:
            # on a atteint le maximum d'itérations, on ne fait plus de M-step
            break

        # M-step: update parameters
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        l_prev = l

    # E-step finale pour avoir g et l cohérents avec les derniers paramètres
    g, l = expectation(X, pi, m, S)
    if g is None or l is None:
        return None, None, None, None, None

    # si la dernière itération n'est pas multiple de 10, il faut encore imprimer
    if verbose and (i % 10 != 0):
        print("Log Likelihood after {} iterations: {}".format(
            i, round(l, 5)))

    return pi, m, S, g, l
