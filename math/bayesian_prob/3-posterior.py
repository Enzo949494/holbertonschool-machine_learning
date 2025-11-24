#!/usr/bin/env python3
"""
Module for calculating posterior probability of hypothetical probabilities.
"""

import numpy as np


def posterior(x, n, P, Pr):
    """
    Calculates the posterior probability for the various hypothetical
    probabilities of developing severe side effects given the data.

    Args:
        x: number of patients that develop severe side effects
        n: total number of patients observed
        P: 1D numpy.ndarray containing hypothetical probabilities
        Pr: 1D numpy.ndarray containing prior beliefs of P

    Returns:
        1D numpy.ndarray containing the posterior probability of each
        probability in P given x and n, respectively
    """

    # Validate n
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")

    # Validate x
    if not isinstance(x, (int, np.integer)) or x < 0:
        raise ValueError("x must be an integer that is greater than or "
                         "equal to 0")

    # Validate x <= n
    if x > n:
        raise ValueError("x cannot be greater than n")

    # Validate P is 1D numpy.ndarray
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # Validate Pr is numpy.ndarray with same shape as P
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape "
                        "as P")

    # Validate all values in P are in [0, 1]
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Validate all values in Pr are in [0, 1]
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    # Validate Pr sums to 1
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate binomial coefficient C(n, x) using factorial
    binomial_coef = np.math.factorial(n) / (np.math.factorial(x) *
                                            np.math.factorial(n - x))

    # Calculate likelihood: C(n,x) * p^x * (1-p)^(n-x)
    likelihood = binomial_coef * (P ** x) * ((1 - P) ** (n - x))

    # Calculate intersection: likelihood * prior
    intersection = likelihood * Pr

    # Calculate marginal probability: sum of intersection
    marginal_prob = np.sum(intersection)

    # Calculate posterior: intersection / marginal probability
    posterior_prob = intersection / marginal_prob

    return posterior_prob