#!/usr/bin/env python3
"""
Module for Multivariate Normal distribution.

This module provides a class to represent and work with multivariate
normal distributions.
"""

import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution.

    Attributes:
        mean: numpy.ndarray of shape (d, 1) containing the mean of data
        cov: numpy.ndarray of shape (d, d) containing the covariance matrix
    """

    def __init__(self, data):
        """
        Initialize a MultiNormal distribution.

        Args:
            data: numpy.ndarray of shape (d, n) containing the data set
                  n is the number of data points
                  d is the number of dimensions in each data point

        Raises:
            TypeError: if data is not a 2D numpy.ndarray
            ValueError: if n is less than 2
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Calculate mean: shape (d, 1)
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Calculate covariance: shape (d, d)
        # Cov = (1/(n-1)) * (X - mean) * (X - mean)^T
        data_centered = data - self.mean
        self.cov = np.dot(data_centered, data_centered.T) / (n - 1)
