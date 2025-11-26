#!/usr/bin/env python3

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    
    cumsum = np.cumsum(S ** 2)
    cumsum = cumsum / cumsum[-1]
    
    nd = np.argmax(cumsum >= var) + 2
    
    return Vt[:nd].T