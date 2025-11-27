#!/usr/bin/env python3
"""
PCA transformation that projects data to a specific dimensionality.
"""

import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset and projects it to a new dimensionality.
    
    Args:
        X (numpy.ndarray): Input data of shape (n, d) where:
            - n is the number of data points
            - d is the number of dimensions/features per data point
        ndim (int): The new dimensionality of the transformed data.
            Must be less than or equal to d.
    
    Returns:
        numpy.ndarray: Transformed data T of shape (n, ndim) containing 
            the data projected onto the first ndim principal components.
    
    Notes:
        - The input data X should be centered (mean = 0) before calling 
          this function for best results
        - Uses Singular Value Decomposition (SVD) for computation
        - The transformation matrix is derived from the right singular vectors
    
    Example:
        >>> X = np.random.randn(100, 50)  # 100 samples, 50 features
        >>> X_centered = X - np.mean(X, axis=0)
        >>> T = pca(X_centered, 10)  # Project to 10 dimensions
        >>> print(T.shape)  # (100, 10)
    """
    # Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Vt contains the principal components as rows
    # Take the first ndim rows and transpose to get (d, ndim) matrix
    W = Vt[:ndim].T
    
    # Project X onto the first ndim principal components
    # T = X @ W gives shape (n, ndim)
    T = np.matmul(X, W)
    
    return T