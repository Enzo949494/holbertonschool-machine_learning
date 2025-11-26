#!/usr/bin/env python3

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.
    
    Args:
        X: numpy.ndarray of shape (n, d)
        var: fraction of variance to maintain (default 0.95)
    
    Returns:
        W: numpy.ndarray of shape (d, nd) with transformation weights
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    
    cumsum = np.cumsum(S ** 2)
    cumsum = cumsum / cumsum[-1]
    
    # Find the number of components needed to maintain var fraction
    nd = np.argmax(cumsum >= var) + 1
    
    return Vt[:nd].T