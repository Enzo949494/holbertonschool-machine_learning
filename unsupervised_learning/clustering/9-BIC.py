#!/usr/bin/env python3
"""
Module finding the best number of clusters using BIC.
"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using BIC.

    X: np.ndarray of shape (n, d)
    kmin: minimum number of clusters to check (inclusive)
    kmax: maximum number of clusters to check (inclusive)
    iterations: maximum number of EM iterations
    tol: tolerance on log-likelihood change for EM
    verbose: if True, EM may print its logs

    Returns:
        best_k: best value for k based on BIC
        best_result: (pi, m, S) for best_k
        l: np.ndarray of shape (kmax - kmin + 1,) log-likelihoods
        b: np.ndarray of shape (kmax - kmin + 1,) BIC values
        or (None, None, None, None) on failure
    """
    # validation de X
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None

    # validation de kmin
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None

    # validation iterations / tol / verbose
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None

    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape

    # gestion de kmax par défaut
    if kmax is None:
        kmax = n

    # validation de kmax et cohérence avec kmin et n
    if not isinstance(kmax, int) or kmax <= 0:
        return None, None, None, None

    if kmax < kmin or kmin > n or kmax > n:
        return None, None, None, None

    # allocation des tableaux de résultats
    l = np.zeros(kmax - kmin + 1)
    b = np.zeros(kmax - kmin + 1)

    best_bic = np.inf
    best_k = None
    best_result = None

    # boucle sur les valeurs de k
    for idx, k in enumerate(range(kmin, kmax + 1)):
        pi, m, S, g, l_val = expectation_maximization(
            X, k, iterations=iterations, tol=tol, verbose=verbose)

        # si EM échoue pour un k, on considère que l'appel est un échec
        if (pi is None or m is None or S is None or
                g is None or l_val is None):
            return None, None, None, None

        l[idx] = l_val

        # nombre de paramètres :
        # pi : k - 1
        # m : k * d
        # S : k * d(d + 1) / 2 (matrices symétriques)
        p = (k - 1) + k * d + k * d * (d + 1) // 2

        # BIC = p * ln(n) - 2 * l
        b[idx] = p * np.log(n) - 2 * l_val

        # garder le plus petit BIC
        if b[idx] < best_bic:
            best_bic = b[idx]
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, l, b
