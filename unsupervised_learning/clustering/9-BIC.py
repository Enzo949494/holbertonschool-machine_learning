#!/usr/bin/env python3
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using BIC."""
    # Validation X
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None

    # Validation kmin
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None

    # Validation iterations / tol / verbose
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None

    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None
    

    n, d = X.shape

    if kmax is None or not isinstance(kmax, int) or kmax <= 0:
        return None, None, None, None

    if kmax < kmin or kmin > n or kmax > n:
        return None, None, None, None


    # Contrainte demandée par le checker
    if kmax < kmin:
        return None, None, None, None

    # Optionnel : on peut aussi refuser kmax > n si l’école le veut
    # (ça ne devrait pas être déclencheur dans ce test mais c’est cohérent)
    if kmin > n or kmax > n:
        return None, None, None, None

    # Allocation des tableaux résultats
    l = np.zeros(kmax - kmin + 1)
    b = np.zeros(kmax - kmin + 1)

    best_bic = np.inf
    best_k = None
    best_result = None

    # Boucle sur k
    for idx, k in enumerate(range(kmin, kmax + 1)):
        pi, m, S, g, l_val = expectation_maximization(
            X, k, iterations=iterations, tol=tol, verbose=verbose)

        if (pi is None or m is None or S is None or
                g is None or l_val is None):
            return None, None, None, None

        l[idx] = l_val

        # Nombre de paramètres
        p = (k - 1) + k * d + k * d * (d + 1) // 2

        # BIC
        b[idx] = p * np.log(n) - 2 * l_val

        if b[idx] < best_bic:
            best_bic = b[idx]
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, l, b
