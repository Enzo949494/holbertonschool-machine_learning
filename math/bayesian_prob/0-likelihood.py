#!/usr/bin/env python3

import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining data given
    various hypothetical probabilities.

    Args:
        x: number of patients that develop severe side effects
        n: total number of patients observed
        P: 1D numpy.ndarray containing hypothetical probabilities

    Returns:
        1D numpy.ndarray containing the likelihood for each probability in P
    """

    # Validate n
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")

    # Validate x
    if not isinstance(x, (int, np.integer)) or x < 0:
        raise ValueError("x must be an integer that is greater than or"
                         "equal to 0")

    # Validate x <= n
    if x > n:
        raise ValueError("x cannot be greater than n")

    # Validate P is 1D numpy.ndarray
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # Validate all values in P are in [0, 1]
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Calculate binomial coefficient C(n, x) using factorial
    binomial_coef = np.math.factorial(n) / (np.math.factorial(x) *
                                            np.math.factorial(n - x))

    # Calculate likelihood for each probability: C(n,x) * p^x * (1-p)^(n-x)
    likelihoods = binomial_coef * (P ** x) * ((1 - P) ** (n - x))

    return likelihoods
