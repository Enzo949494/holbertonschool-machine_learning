#!/usr/bin/env python3
"""
Principal Component Analysis (PCA) implementation for dimensionality reduction.
"""

import numpy as np


def pca(X, var=0.95):
    """
    Performs Principal Component Analysis on a dataset.
    
    Principal Component Analysis (PCA) is a dimensionality reduction technique
    that transforms data into a new coordinate system where the greatest variance
    by any projection of the data lies on the first coordinate (first principal 
    component), the second greatest variance on the second coordinate, and so on.
    
    Args:
        X (numpy.ndarray): Input data of shape (n, d) where:
            - n is the number of data points
            - d is the number of dimensions/features per data point
            - All dimensions must have a mean of 0 across all data points
        var (float, optional): The fraction of the variance that the PCA 
            transformation should maintain. Must be between 0 and 1.
            Default is 0.95 (maintains 95% of variance).
    
    Returns:
        numpy.ndarray: Weight matrix W of shape (d, nd) where nd is the new 
            dimensionality. This matrix can be used to project data into the 
            lower-dimensional space: T = X @ W
    
    Notes:
        - Uses Singular Value Decomposition (SVD) for computation
        - The input data X should be centered (mean = 0) before calling this function
        - The returned matrix W contains the principal component vectors as columns
    
    Example:
        >>> X = np.random.randn(100, 50)  # 100 samples, 50 features
        >>> X_centered = X - np.mean(X, axis=0)
        >>> W = pca(X_centered, var=0.95)
        >>> X_transformed = X_centered @ W  # Project to lower dimensions
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    
    cumsum = np.cumsum(S ** 2)
    cumsum = cumsum / cumsum[-1]
    
    nd = np.argmax(cumsum >= var) + 2
    
    return Vt[:nd].T
