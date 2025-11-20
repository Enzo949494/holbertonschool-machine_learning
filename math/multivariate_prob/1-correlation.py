#!/usr/bin/env python3
"""
Module for calculating correlation matrix from covariance matrix.

This module provides functionality to compute the correlation matrix
from a covariance matrix.
"""

import numpy as np


def correlation(C):
    """
    Calculate a correlation matrix from a covariance matrix.

    Args:
        C: numpy.ndarray of shape (d, d) containing a covariance matrix
           d is the number of dimensions

    Returns:
        numpy.ndarray of shape (d, d) containing the correlation matrix

    Raises:
        TypeError: if C is not a numpy.ndarray
        ValueError: if C does not have shape (d, d)
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # Extract standard deviations from diagonal (variance = std^2)
    std = np.sqrt(np.diag(C))

    # Create correlation matrix: corr[i,j] = cov[i,j] / (std[i] * std[j])
    correlation_matrix = C / np.outer(std, std)

    return correlation_matrix
