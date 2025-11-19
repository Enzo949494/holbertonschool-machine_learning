#!/usr/bin/env python3
"""
Module for calculating matrix definiteness.
"""

import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix.

    Args:
        matrix: A numpy.ndarray of shape (n, n)
                whose definiteness should be calculated

    Returns:
        The string describing the definiteness of the matrix,
        or None if invalid

    Raises:
        TypeError: If matrix is not a numpy.ndarray
    """
    # Check if matrix is a numpy.ndarray
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Check if matrix is valid (2D and square)
    if matrix.ndim != 2:
        return None

    if matrix.shape[0] != matrix.shape[1]:
        return None

    if matrix.shape[0] == 0:
        return None

    # Check if matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        return None

    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)

    # Round eigenvalues to handle floating point errors
    eigenvalues = np.round(eigenvalues, decimals=10)

    # Count positive, negative, and zero eigenvalues
    positive = np.sum(eigenvalues > 0)
    negative = np.sum(eigenvalues < 0)
    zero = np.sum(eigenvalues == 0)

    n = len(eigenvalues)

    # Determine definiteness based on eigenvalues
    if negative == 0 and zero == 0:
        return "Positive definite"
    elif negative == 0 and positive > 0:
        return "Positive semi-definite"
    elif positive == 0 and zero == 0:
        return "Negative definite"
    elif positive == 0 and negative > 0:
        return "Negative semi-definite"
    else:
        return "Indefinite"
