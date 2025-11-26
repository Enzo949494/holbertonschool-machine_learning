import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    
    cumsum = np.cumsum(S ** 2)
    cumsum = cumsum / cumsum[-1]
    
    # Trouver le nombre de composantes pour var
    nd_var = np.argmax(cumsum >= var) + 1
    
    # Compter les composantes significatives (S > 1e-12)
    nd_sig = np.sum(S > 1e-12)
    
    # Prendre le maximum des deux
    nd = max(nd_var, nd_sig)
    
    return Vt[:nd].T