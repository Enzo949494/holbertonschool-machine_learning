#!/usr/bin/env python3
"""
Module for calculating mean and covariance of a dataset.

This module provides functionality to compute the mean vector and covariance
matrix of a multivariate dataset represented as a 2D numpy array.
"""

import numpy as np


def mean_cov(X):
    """
    Calculate the mean and covariance of a data set.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
           n is the number of data points
           d is the number of dimensions in each data point

    Returns:
        mean: numpy.ndarray of shape (1, d) containing the mean of the data set
        cov: numpy.ndarray of shape (d, d) containing the covariance matrix

    Raises:
        TypeError: if X is not a 2D numpy.ndarray
        ValueError: if n is less than 2
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n = X.shape[0]
    if n < 2:
        raise ValueError("X must contain multiple data points")

    # Calculate mean
    mean = np.mean(X, axis=0, keepdims=True)

    # Calculate covariance manually
    # Cov = (1/n) * (X - mean)^T * (X - mean)
    X_centered = X - mean
    cov = np.dot(X_centered.T, X_centered) / n

    return mean, cov
