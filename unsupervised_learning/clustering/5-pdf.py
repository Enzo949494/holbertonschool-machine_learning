#!/usr/bin/env python3
"""
Module calculating Gaussian probability density function.

This module provides functionality to calculate the PDF of a
multivariate Gaussian distribution given data points.
"""

import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution.

    Args:
        X: numpy.ndarray of shape (n, d) - data points
        m: numpy.ndarray of shape (d,) - mean of the distribution
        S: numpy.ndarray of shape (d, d) - covariance matrix

    Returns:
        P: numpy.ndarray of shape (n,) - PDF values each data point
           or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None

    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None

    if X.shape[1] != m.shape[0] or S.shape[0] != S.shape[1]:
        return None

    if X.shape[1] != S.shape[0]:
        return None

    n, d = X.shape

    # Calculate the numerator: (X - m) @ S^-1 @ (X - m)^T
    X_minus_m = X - m
    S_inv = np.linalg.inv(S)

    # Calculate Mahalanobis distance squared all points
    numerator = np.sum(X_minus_m @ S_inv * X_minus_m, axis=1)

    # Calculate determinant and normalization factor
    det_S = np.linalg.det(S)

    if det_S <= 0:
        return None

    norm_factor = 1 / (np.sqrt((2 * np.pi) ** d * det_S))

    # Calculate PDF: norm_factor * exp(-0.5 * numerator)
    P = norm_factor * np.exp(-0.5 * numerator)

    # Ensure minimum value of 1e-300
    P = np.maximum(P, 1e-300)

    return P
