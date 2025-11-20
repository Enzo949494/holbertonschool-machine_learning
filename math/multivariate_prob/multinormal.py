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

    def pdf(self, x):
        """
        Calculate the PDF at a data point.

        Args:
            x: numpy.ndarray of shape (d, 1) containing the data point
               whose PDF should be calculated

        Returns:
            float: the value of the PDF at point x

        Raises:
            TypeError: if x is not a numpy.ndarray
            ValueError: if x is not of shape (d, 1)
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")

        # PDF formula for multivariate normal:
        # P(x) = (1 / ((2π)^(d/2) * |Σ|^(1/2))) *
        # exp(-0.5 * (x-μ)^T * Σ^(-1) * (x-μ))

        # Calculate determinant and inverse of covariance matrix
        det_cov = np.linalg.det(self.cov)
        inv_cov = np.linalg.inv(self.cov)

        # Calculate the difference: (x - mean)
        diff = x - self.mean

        # Calculate the exponent: -0.5 * (x-μ)^T * Σ^(-1) * (x-μ)
        exponent = -0.5 * np.dot(diff.T, np.dot(inv_cov, diff))[0, 0]

        # Calculate the normalization constant: 1 / ((2π)^(d/2) * |Σ|^(1/2))
        numerator = 1
        denominator = np.sqrt((2 * np.pi) ** d * det_cov)

        # Calculate and return the PDF
        pdf_value = (numerator / denominator) * np.exp(exponent)

        return float(pdf_value)
